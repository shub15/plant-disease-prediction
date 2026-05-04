# Plant Disease Detector

A modern, AI-powered system designed for **Raspberry Pi 5** that detects plant leaves and identifies diseases in real-time. The project uses a 2-stage pipeline (YOLOv8 + MobileNetV2) to provide accurate diagnoses with a sleek, mobile-responsive web interface.

---

## Features

- **2-Stage AI Pipeline**: 
    - **Stage 1 (YOLOv8)**: Detects and crops leaves from the camera frame.
    - **Stage 2 (MobileNetV2)**: Classifies the specific disease from the cropped leaf.
- **Species-Aware Detection**: Identifies 46 different plant species (Tomato, Grape, Mango, etc.) before diagnosing the disease.
- **Real-time Streaming**: Live video feed with annotated bounding boxes and FPS overlay.
- **Mobile Responsive**: Optimized for smartphones, tablets, and Raspberry Pi touchscreens.
- **Modern UI**: Built with the premium Cohere design system (dark mode, glassmorphism).
- **Flexible Input**: Supports live camera (USB or Pi Camera) and image uploads.

---

## 🚀 Quick Start (Raspberry Pi)

### 1. Setup Environment
```bash
# Clone the repository
cd plant-disease-prediction

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Server
```bash
python flask_app.py
```

### 3. Access the UI
- **On Pi**: Open `http://localhost:8080`
- **On Phone/Laptop**: Open `http://<your-pi-ip>:8080`

---

##  Mobile Access

The interface is designed to be "mobile-first":
1. Find your Pi's IP address: `hostname -I`
2. Connect your phone to the same Wi-Fi as the Pi.
3. Open your mobile browser and navigate to the IP address on port 8080.

---

##  UI Controls

- **Start/Stop Camera**: Toggle the live video feed.
- **Save Snapshot**: Download the current annotated frame as a JPEG.
- **Mode Selector**: Switch between **Live Camera** and **Upload Image** mode.
- **Confidence Slider**: Adjust how "sure" the AI needs to be to show a detection.
- **Skip YOLO**: A faster mode that runs classification on the full frame (good for low-power situations).

---

##  Project Structure

- `flask_app.py`: The main web server (Flask + Socket.IO).
- `detect.py`: The core AI logic and inference pipeline.
- `best.pt`: The specialized YOLOv8 model for plant detection.
- `mobilenetv2_plant.pth`: The disease classification model.
- `templates/` & `static/`: Frontend HTML, CSS, and JavaScript.

---

## Performance on RPi 5

| Mode | Speed | Notes |
|------|-------|-------|
| YOLO + MobileNetV2 | 2-3 sec/frame | Highest accuracy, identifies species |
| MobileNetV2 only | 0.5-1 sec/frame | Faster, uses "Skip YOLO" toggle |

---

## Troubleshooting

- **Camera not showing?** Ensure your user is in the video group: `sudo usermod -aG video $USER` (reboot after).
- **Port 8080 busy?** You can change the port in `flask_app.py` or kill the existing process: `fuser -k 8080/tcp`.
- **Sluggish Video?** The app processes every 5th frame to keep the UI smooth. Ensure you have a stable Wi-Fi connection if viewing from a phone.

---

##  Credits & Models

- **Leaf Detection**: YOLOv8 (nano) trained on plant datasets.
- **Disease Classification**: MobileNetV2 trained on the PlantVillage dataset (38 classes).
- **Design**: Inspired by the Cohere enterprise design palette.