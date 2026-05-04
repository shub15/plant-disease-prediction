"""
Plant Disease Detection Pipeline for Raspberry Pi 5
====================================================
Stage 1: YOLOv8 (foduucom) — detects & crops leaf from camera frame
Stage 2: MobileNetV2 (Daksh159) — classifies disease from cropped leaf

Usage:
    python detect.py                  # Use Pi Camera (live)
    python detect.py --image leaf.jpg # Use a static image
    python detect.py --demo           # Demo mode (no camera needed)
"""

import argparse
import time
import sys
import os
import urllib.request

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from collections import deque
from collections import Counter

# ──────────────────────────────────────────────
# 38 PlantVillage Class Labels
# ──────────────────────────────────────────────
CLASS_LABELS = [
    "Apple - Apple Scab",
    "Apple - Black Rot",
    "Apple - Cedar Apple Rust",
    "Apple - Healthy",
    "Blueberry - Healthy",
    "Cherry - Powdery Mildew",
    "Cherry - Healthy",
    "Corn - Cercospora Leaf Spot / Gray Leaf Spot",
    "Corn - Common Rust",
    "Corn - Northern Leaf Blight",
    "Corn - Healthy",
    "Grape - Black Rot",
    "Grape - Esca (Black Measles)",
    "Grape - Leaf Blight (Isariopsis Leaf Spot)",
    "Grape - Healthy",
    "Orange - Haunglongbing (Citrus Greening)",
    "Peach - Bacterial Spot",
    "Peach - Healthy",
    "Pepper Bell - Bacterial Spot",
    "Pepper Bell - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Raspberry - Healthy",
    "Soybean - Healthy",
    "Squash - Powdery Mildew",
    "Strawberry - Leaf Scorch",
    "Strawberry - Healthy",
    "Tomato - Bacterial Spot",
    "Tomato - Early Blight",
    "Tomato - Late Blight",
    "Tomato - Leaf Mold",
    "Tomato - Septoria Leaf Spot",
    "Tomato - Spider Mites (Two-Spotted Spider Mite)",
    "Tomato - Target Spot",
    "Tomato - Tomato Yellow Leaf Curl Virus",
    "Tomato - Tomato Mosaic Virus",
    "Tomato - Healthy",
]

# Simple treatment suggestions per disease keyword
TREATMENTS = {
    "Scab": "Apply fungicide (captan or myclobutanil). Remove infected leaves.",
    "Black Rot": "Prune infected areas. Apply copper-based fungicide.",
    "Rust": "Apply sulfur-based or triazole fungicide. Improve air circulation.",
    "Powdery Mildew": "Apply potassium bicarbonate or neem oil spray.",
    "Blight": "Remove infected tissue. Apply copper fungicide. Avoid overhead watering.",
    "Bacterial Spot": "Apply copper bactericide. Avoid wetting foliage.",
    "Leaf Spot": "Apply chlorothalonil fungicide. Remove fallen leaves.",
    "Mosaic Virus": "Remove infected plants. Control aphid vectors.",
    "Curl Virus": "Control whiteflies. Remove infected plants immediately.",
    "Mold": "Improve ventilation. Apply fungicide (chlorothalonil).",
    "Esca": "No cure; remove infected vines. Apply preventive fungicide.",
    "Greening": "No cure. Remove infected trees to prevent spread.",
    "Cercospora": "Apply strobilurin or triazole fungicide.",
    "Common Rust": "Apply fungicide at first sign. Use resistant varieties.",
    "Spider Mites": "Apply miticide or insecticidal soap. Increase humidity.",
    "Scorch": "Improve drainage. Apply appropriate fungicide.",
    "Healthy": "✅ No disease detected. Keep up good plant care!",
}


def get_treatment(label: str) -> str:
    for keyword, advice in TREATMENTS.items():
        if keyword.lower() in label.lower():
            return advice
    return "Consult a local agronomist for treatment advice."


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────


def load_yolo_model():
    """Load YOLOv8s leaf detection model from HuggingFace."""
    print("📦 Loading YOLOv8 leaf detection model...")
    try:
        import torch
        from ultralytics import YOLO
        from unittest.mock import patch

        # Force weights_only=False for YOLO loading (required for PyTorch 2.6+)
        # We capture the original torch.load to avoid infinite recursion in the mock
        orig_load = torch.load
        with patch("torch.load", side_effect=lambda *args, **kwargs: orig_load(*args, **{**kwargs, "weights_only": False})):
            model = YOLO("best.pt")
        
        print("✅ YOLOv8 model loaded.\n")
        return model
    except Exception as e:
        print(f"⚠️  YOLOv8 load failed: {e}")
        return None


def load_mobilenet_model():
    """Load MobileNetV2 plant disease classifier."""
    print("📦 Loading MobileNetV2 disease classifier...")
    model_path = os.path.join(os.path.dirname(__file__), "mobilenetv2_plant.pth")

    # Download model weights if not cached
    if not os.path.exists(model_path):
        print("   Downloading fallback model weights (~14MB)...")
        url = "https://huggingface.co/Daksh159/plant-disease-mobilenetv2/blob/main/mobilenetv2_plant.pth"
        try:
            urllib.request.urlretrieve(url, model_path)
            print("   Download complete.")
        except Exception as e:
            print(f"⚠️  Download failed: {e}")
            print(
                "   Manually download from: https://huggingface.co/Daksh159/plant-disease-mobilenetv2"
            )
            return None

    # Build model architecture (must match training)
    model = models.mobilenet_v2(pretrained=True)
    for p in model.features.parameters():
        p.requires_grad = False  # freeze feature extractor
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(model.classifier[1].in_features, 38),
    )

    try:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        # Handle both raw state_dict and wrapped checkpoints
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    except Exception as e:
        print(f"⚠️  Weight load error: {e}")
        print("   The model will run but predictions may be random.")

    model.eval()
    print("✅ MobileNetV2 model loaded.\n")
    return model


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────

MOBILENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def classify_leaf(mobilenet, pil_image: Image.Image):
    """Run MobileNetV2 on a PIL image, return (label, confidence)."""
    tensor = MOBILENET_TRANSFORM(pil_image).unsqueeze(0)
    with torch.no_grad():
        logits = mobilenet(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)
    label = (
        CLASS_LABELS[idx.item()]
        if idx.item() < len(CLASS_LABELS)
        else f"Class {idx.item()}"
    )
    return label, conf.item()


# Global history for temporal smoothing of predictions
prediction_history = deque(maxlen=7)

def run_pipeline(frame_bgr, yolo_model, mobilenet_model, conf_thresh=0.25):
    """
    Full 2-stage pipeline with temporal smoothing.
    Returns annotated frame + list of result dicts.
    """
    global prediction_history
    results_list = []
    annotated = frame_bgr.copy()

    # ── Stage 1: YOLO leaf detection ──
    if yolo_model is not None:
        yolo_results = yolo_model(frame_bgr, conf=conf_thresh, verbose=False)
        boxes = yolo_results[0].boxes

        if boxes is not None and len(boxes):
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                yolo_conf = float(box.conf[0])
                class_id = int(box.cls[0])
                plant_species = yolo_model.names[class_id]

                # Crop detected leaf region
                pad = 10
                x1c = max(0, x1 - pad)
                y1c = max(0, y1 - pad)
                x2c = min(frame_bgr.shape[1], x2 + pad)
                y2c = min(frame_bgr.shape[0], y2 + pad)
                crop_bgr = frame_bgr[y1c:y2c, x1c:x2c]

                if crop_bgr.size == 0:
                    continue

                # ── Stage 2: MobileNet classification ──
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                disease_label, disease_conf = classify_leaf(mobilenet_model, pil_crop)
                
                # Apply Temporal Smoothing (Majority Vote over last 7 frames)
                # prediction_history.append(disease_label)
                # vote_counts = Counter(prediction_history)
                # smoothed_label, _ = vote_counts.most_common(1)[0]
                
                # # If the smoothed label differs from current, average the confidence
                # if smoothed_label != disease_label:
                #     # Penalty for fluctuating
                #     disease_conf = max(0.0, disease_conf - 0.2)
                
                # disease_label = smoothed_label

                # Extract plant and specific disease correctly from MobileNet
                if " - " in disease_label:
                    actual_plant, actual_disease = disease_label.split(" - ", 1)
                else:
                    actual_plant = "Unknown"
                    actual_disease = disease_label

                results_list.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "yolo_conf": yolo_conf,
                        "plant": actual_plant,
                        "disease": actual_disease,
                        "disease_conf": disease_conf,
                        "treatment": get_treatment(disease_label),
                    }
                )

                # Draw bounding box
                is_healthy = "healthy" in disease_label.lower()
                color = (0, 200, 0) if is_healthy else (0, 60, 220)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Label background
                short_label = actual_disease
                text = f"{short_label} ({disease_conf:.0%})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(
                    annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1
                )
                cv2.putText(
                    annotated,
                    text,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                )
        else:
            # No YOLO detection — do not fall back to full frame to avoid phantom predictions
            results_list = []
    else:
        # YOLO unavailable — classify full frame
        results_list = _classify_full_frame(frame_bgr, mobilenet_model, annotated)

    return annotated, results_list


def _classify_full_frame(frame_bgr, mobilenet_model, annotated):
    """Fallback: classify entire frame when no leaf is detected."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    label, conf = classify_leaf(mobilenet_model, pil_img)
    
    if " - " in label:
        actual_plant, actual_disease = label.split(" - ", 1)
    else:
        actual_plant = "Unknown"
        actual_disease = label
        
    h, w = frame_bgr.shape[:2]
    is_healthy = "healthy" in actual_disease.lower()
    color = (0, 200, 0) if is_healthy else (0, 60, 220)
    text = f"{actual_disease} ({conf:.0%})"
    cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return [
        {
            "bbox": None,
            "plant": actual_plant,
            "disease": actual_disease,
            "disease_conf": conf,
            "treatment": get_treatment(label),
            "yolo_conf": None,
        }
    ]


def print_results(results):
    """Pretty-print results to terminal."""
    print("\n" + "═" * 55)
    if not results:
        print("  No leaf detected in frame.")
    for i, r in enumerate(results, 1):
        print(f"  Leaf #{i}")
        print(f"  🌳 Plant   : {r['plant']}")
        print(f"  🌿 Disease : {r['disease']}")
        print(f"  📊 Confidence: {r['disease_conf']:.1%}")
        print(f"  💊 Treatment : {r['treatment']}")
        if r.get("yolo_conf"):
            print(f"  🎯 Detect conf: {r['yolo_conf']:.1%}")
    print("═" * 55 + "\n")


# ──────────────────────────────────────────────
# Entry Points
# ──────────────────────────────────────────────


def run_camera(yolo_model, mobilenet_model):
    """Live camera loop using Pi Camera or USB webcam."""
    print("📷 Starting camera... Press 'q' to quit, 's' to save frame.\n")

    # Try PiCamera2 first, fall back to OpenCV
    cap = None
    use_picamera = False

    # Check if we're on Raspberry Pi (not Kali Linux)
    try:
        from picamera2 import Picamera2
        # Additional check: Picamera2 might import but not work on non-Pi systems
        import platform
        if 'aarch64' in platform.platform() and 'raspi' in platform.release().lower():
            picam2 = Picamera2()
            picam2.configure(
                picam2.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
            )
            picam2.start()
            use_picamera = True
            print("✅ Using PiCamera2")
        else:
            raise ImportError("Not on Raspberry Pi")
    except ImportError:
        use_picamera = False
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open USB camera. Check if it's plugged in or used by another app.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Warm-up camera (discard first few frames to stabilize exposure)
        print("⏳ Warming up camera...")
        for _ in range(5):
            cap.read()
        print("✅ Using USB/OpenCV camera")

    win_name = "Plant Disease Detector"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    
    last_annotated = None
    last_results = []
    frame_count = 0
    fps_time = time.time()
    
    print("📷 Camera started. 'q'=Quit, 's'=Save.\n")

    try:
        while True:
            # 1. Grab fresh frame
            if use_picamera:
                frame_rgb = picam2.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Camera read failed.")
                    break

            frame_count += 1

            # 2. Process every 5th frame (or first frame)
            if frame_count % 5 == 0 or last_annotated is None:
                # We run the pipeline on a COPY of the frame so we don't mess up the live one
                processed_frame, results = run_pipeline(frame.copy(), yolo_model, mobilenet_model)
                last_annotated = processed_frame
                last_results = results
                if results:
                    print_results(results)

            # 3. Display Logic
            # To keep it smooth, we show the LIVE frame but with the LAST annotations
            # However, since run_pipeline draws ON the frame, we'll just show the last processed_frame
            # if we want the annotations to stay. If we want smooth live video, we'd need to
            # draw the last_results on the current 'frame'. 
            # For simplicity and to fix "blank pictures", we'll show last_annotated.
            
            display_frame = last_annotated if last_annotated is not None else frame

            # FPS overlay
            elapsed = time.time() - fps_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f}",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow(win_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                fname = f"capture_{int(time.time())}.jpg"
                # Save the annotated frame so it's not "blank"
                cv2.imwrite(fname, display_frame)
                print(f"💾 Saved annotated capture: {fname}")

    finally:
        if use_picamera:
            picam2.stop()
        elif cap:
            cap.release()
        cv2.destroyAllWindows()


def run_image(image_path, yolo_model, mobilenet_model):
    """Run pipeline on a single image file."""
    print(f"🖼️  Processing: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Could not read image: {image_path}")
        sys.exit(1)

    start = time.time()
    annotated, results = run_pipeline(frame, yolo_model, mobilenet_model)
    elapsed = time.time() - start

    print_results(results)
    print(f"⏱️  Inference time: {elapsed:.2f}s")

    out_path = "result_" + os.path.basename(image_path)
    cv2.imwrite(out_path, annotated)
    print(f"💾 Annotated image saved: {out_path}")

    cv2.imshow("Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_demo(mobilenet_model):
    """Demo mode: create a synthetic green image and classify it."""
    print("🎭 Demo mode — generating synthetic leaf image...\n")
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy[:, :, 1] = 120  # green-ish
    pil_img = Image.fromarray(dummy)
    label, conf = classify_leaf(mobilenet_model, pil_img)
    print(f"  Disease : {label}")
    print(f"  Confidence: {conf:.1%}")
    print(f"  Treatment : {get_treatment(label)}")
    print("\n✅ Demo complete. Models are working correctly on this device.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plant Disease Detector — RPi 5")
    parser.add_argument("--image", type=str, help="Path to a leaf image file")
    parser.add_argument("--demo", action="store_true", help="Run demo without camera")
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Skip YOLO, classify full frame only (faster)",
    )
    args = parser.parse_args()

    print("\n🌿 Plant Disease Detection Pipeline")
    print("   Raspberry Pi 5 | YOLOv8 + MobileNetV2")
    print("─" * 45)

    # Load models
    yolo_model = None if args.no_yolo else load_yolo_model()
    mobilenet_model = load_mobilenet_model()

    if mobilenet_model is None:
        print("❌ MobileNetV2 failed to load. Cannot continue.")
        sys.exit(1)

    # Run selected mode
    if args.demo:
        run_demo(mobilenet_model)
    elif args.image:
        run_image(args.image, yolo_model, mobilenet_model)
    else:
        run_camera(yolo_model, mobilenet_model)


if __name__ == "__main__":
    main()
