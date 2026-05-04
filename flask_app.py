#!/usr/bin/env python3
"""
Plant Disease Detector - Flask Web UI
======================================
Flask-based web interface with same design as Streamlit version
Runs on port 8080 with mobile-responsive design
"""

import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
import torch
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import base64
import json

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import pipeline from detect.py
from detect import (
    CLASS_LABELS,
    TREATMENTS,
    load_yolo_model,
    load_mobilenet_model,
    run_pipeline,
    classify_leaf,
    get_treatment
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'plant-disease-detector-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global model instances
yolo_model = None
mobilenet_model = None
models_loaded = False
camera_active = False
camera_cap = None


def initialize_models(use_yolo=True):
    """Initialize AI models on startup."""
    global yolo_model, mobilenet_model, models_loaded

    if models_loaded:
        return True

    try:
        # Load YOLO model if requested
        if use_yolo:
            yolo_model = load_yolo_model()

        # Load MobileNet model
        mobilenet_model = load_mobilenet_model()

        if mobilenet_model is None:
            print("❌ Failed to load MobileNetV2 model")
            return False

        models_loaded = True
        print("✅ Models loaded successfully")
        return True

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False


def generate_frames():
    """Generator function for streaming camera frames."""
    global camera_cap, camera_active

    if camera_cap is None:
        camera_cap = cv2.VideoCapture(0)
        camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warm up camera
        for _ in range(5):
            camera_cap.read()

    frame_count = 0
    fps_time = time.time()
    last_annotated_frame = None

    while camera_active:
        ret, frame = camera_cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every 5th frame
        if frame_count % 5 == 0 or last_annotated_frame is None:
            annotated_frame, results = run_pipeline(frame, yolo_model, mobilenet_model)
            last_annotated_frame = annotated_frame

            # Format results for Socket.IO
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append({
                    'leaf_id': i,
                    'plant': result['plant'],
                    'disease': result['disease'],
                    'confidence': round(result['disease_conf'] * 100, 1),
                    'treatment': result['treatment'],
                    'bbox': result['bbox'],
                    'is_healthy': 'healthy' in result['disease'].lower()
                })

            # Emit results via Socket.IO
            socketio.emit('results', {'results': formatted_results})

        # Add FPS overlay to the frame we're about to send
        display_frame = last_annotated_frame if last_annotated_frame is not None else frame
        
        # We need to copy it if we want to add a unique FPS to every frame without modifying the cached one
        display_frame = display_frame.copy()
        
        elapsed = time.time() - fps_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.1f}",
                   (10, display_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Cleanup
    if camera_cap:
        camera_cap.release()
        camera_cap = None


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/start_camera')
def start_camera():
    """Start camera streaming."""
    global camera_active
    camera_active = True
    return jsonify({'status': 'started'})


@app.route('/stop_camera')
def stop_camera():
    """Stop camera streaming."""
    global camera_active
    camera_active = False
    return jsonify({'status': 'stopped'})


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    global camera_active
    # Automatically start camera if not already active
    if not camera_active:
        camera_active = True
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process a single frame and return results."""
    try:
        # Get image from request
        if 'image' in request.files:
            file = request.files['image']
            frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        elif 'frame' in request.json:
            # Handle base64 encoded frame
            frame_data = request.json['frame']
            nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'No image provided'}), 400

        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Process frame
        annotated_frame, results = run_pipeline(frame, yolo_model, mobilenet_model)

        # Convert annotated frame to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                'leaf_id': i,
                'plant': result['plant'],
                'disease': result['disease'],
                'confidence': round(result['disease_conf'] * 100, 1),
                'treatment': result['treatment'],
                'bbox': result['bbox'],
                'is_healthy': 'healthy' in result['disease'].lower()
            })

        return jsonify({
            'frame': frame_base64,
            'results': formatted_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and processing."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read and process image
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Process frame
        annotated_frame, results = run_pipeline(frame, yolo_model, mobilenet_model)

        # Convert to base64 for display
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Format results
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                'leaf_id': i,
                'plant': result['plant'],
                'disease': result['disease'],
                'confidence': round(result['disease_conf'] * 100, 1),
                'treatment': result['treatment'],
                'bbox': result['bbox'],
                'is_healthy': 'healthy' in result['disease'].lower()
            })

        return jsonify({
            'frame': frame_base64,
            'results': formatted_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    print('Client connected')
    emit('connected', {'data': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')


if __name__ == '__main__':
    # Initialize models
    print("🌿 Plant Disease Detector - Flask Web UI")
    print("Initializing AI models...")

    if initialize_models(use_yolo=True):
        print("✅ Starting Flask server on port 8080")
        socketio.run(app, host='0.0.0.0', port=8080, debug=True, allow_unsafe_werkzeug=True)
    else:
        print("❌ Failed to initialize models. Exiting.")
        sys.exit(1)