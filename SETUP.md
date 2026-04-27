# 🌿 Plant Disease Detector — Raspberry Pi 5 Setup Guide

## What This Does
A 2-stage AI pipeline using your Pi Camera:
- **Stage 1 (YOLOv8):** Detects and draws a bounding box around the leaf in the frame
- **Stage 2 (MobileNetV2):** Classifies the cropped leaf into one of 38 disease categories
- Prints disease name, confidence %, and treatment recommendation

---

## Hardware Required
- Raspberry Pi 5 (8GB) ✅
- Pi Camera Module v2/v3 **or** USB webcam
- MicroSD card (32GB+ recommended)
- Internet connection (for first-time model download only)

---

## Step 1 — Update Your Pi

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git libopencv-dev
```

---

## Step 2 — Create a Virtual Environment

```bash
cd ~
python3 -m venv plant_env
source plant_env/bin/activate
```

> ⚠️ Always activate this environment before running the project:
> `source ~/plant_env/bin/activate`

---

## Step 3 — Clone / Copy Project Files

```bash
mkdir ~/plant_disease_detector
cd ~/plant_disease_detector
# Copy detect.py and requirements.txt here
```

---

## Step 4 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏳ This will take 5–10 minutes on RPi 5. PyTorch ARM wheels are pre-built so no compilation needed.

---

## Step 5 — Enable Camera (if using Pi Camera Module)

```bash
# Enable camera interface
sudo raspi-config
# Go to: Interface Options → Camera → Enable

# Install PiCamera2
pip install picamera2
```

If using a **USB webcam**, skip this step entirely — it works automatically.

---

## Step 6 — Run the Detector

### Option A: Live Camera Feed
```bash
python detect.py
```

### Option B: Test on a Single Image
```bash
python detect.py --image leaf.jpg
```

### Option C: Demo Mode (no camera needed)
```bash
python detect.py --demo
```

### Option D: Faster Mode (skip YOLO, classify full frame)
```bash
python detect.py --no-yolo
```

---

## Controls (while camera is running)
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save current annotated frame as JPEG |

---

## Expected Output (terminal)

```
═══════════════════════════════════════════════════════
  Leaf #1
  🌿 Disease : Tomato - Late Blight
  📊 Confidence: 91.3%
  💊 Treatment : Remove infected tissue. Apply copper fungicide.
  🎯 Detect conf: 87.5%
═══════════════════════════════════════════════════════
```

---

## Performance on RPi 5 (8GB)

| Mode | Approx. Speed |
|------|---------------|
| YOLOv8 + MobileNetV2 (full pipeline) | ~2–3 sec/frame |
| MobileNetV2 only (`--no-yolo`) | ~0.5–1 sec/frame |

> 💡 The pipeline runs every 5th frame in camera mode to keep the display smooth.

---

## Supported Plants & Diseases (38 Classes)

| Plant | Diseases Detected |
|-------|-------------------|
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Corn | Cercospora/Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca, Leaf Blight, Healthy |
| Pepper Bell | Bacterial Spot, Healthy |
| Strawberry | Leaf Scorch, Healthy |
| Peach | Bacterial Spot, Healthy |
| + more | Blueberry, Cherry, Orange, Raspberry, Soybean, Squash |

---

## Troubleshooting

**Camera not found:**
```bash
# Check connected cameras
v4l2-ctl --list-devices
# Try a different index
# Edit detect.py: cv2.VideoCapture(1) instead of (0)
```

**Model download fails:**
- Download manually from: https://huggingface.co/Daksh159/plant-disease-mobilenetv2
- Place the `.pth` file in the same folder as `detect.py` as `mobilenetv2_plant_disease.pth`

**Out of memory:**
```bash
# Reduce swap if needed, or just use --no-yolo mode
python detect.py --no-yolo
```

**torch not found:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Project Structure

```
plant_disease_detector/
├── detect.py               ← Main script
├── requirements.txt        ← Python dependencies
├── SETUP.md                ← This file
└── mobilenetv2_plant_disease.pth  ← Downloaded automatically on first run
```

---

## Model Credits
- **YOLOv8 Leaf Detector:** [foduucom/plant-leaf-detection-and-classification](https://huggingface.co/foduucom/plant-leaf-detection-and-classification) — 46 plant species, mAP 94.6%
- **MobileNetV2 Classifier:** [Daksh159/plant-disease-mobilenetv2](https://huggingface.co/Daksh159/plant-disease-mobilenetv2) — 38 disease classes, PlantVillage dataset
- **Dataset:** PlantVillage (54,000+ leaf images)
