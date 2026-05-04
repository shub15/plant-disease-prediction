// Plant Disease Detector - Flask UI JavaScript

class PlantDiseaseApp {
    constructor() {
        this.currentMode = 'camera';
        this.cameraActive = false;
        this.modelsLoaded = false;
        this.lastResults = [];
        this.videoStream = null;

        this.initializeElements();
        this.bindEvents();
        this.connectWebSocket();
    }

    initializeElements() {
        // Mode elements
        this.modeRadios = document.querySelectorAll('input[name="mode"]');

        // Camera elements
        this.videoFeed = document.getElementById('videoFeed');
        this.videoPlaceholder = document.getElementById('videoPlaceholder');
        this.startCameraBtn = document.getElementById('startCamera');
        this.stopCameraBtn = document.getElementById('stopCamera');
        this.saveSnapshotBtn = document.getElementById('saveSnapshot');

        // Upload elements
        this.imageUpload = document.getElementById('imageUpload');
        this.uploadArea = document.getElementById('uploadArea');
        this.browseButton = document.getElementById('browseButton');
        this.uploadControls = document.getElementById('uploadControls');
        this.processedImage = document.getElementById('processedImage');
        this.uploadedImageContainer = document.getElementById('uploadedImage');

        // Settings elements
        this.noYoloCheckbox = document.getElementById('noYolo');
        this.confidenceSlider = document.getElementById('confidence');
        this.confidenceValue = document.querySelector('.range-value');

        // Results elements
        this.resultsContainer = document.getElementById('resultsContainer');
        this.summaryStats = document.getElementById('summaryStats');
        this.totalLeavesEl = document.getElementById('totalLeaves');
        this.healthyCountEl = document.getElementById('healthyCount');
        this.diseasedCountEl = document.getElementById('diseasedCount');

        // Loading overlay
        this.loadingOverlay = document.getElementById('loadingOverlay');
    }

    bindEvents() {
        // Mode switching
        this.modeRadios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.switchMode(e.target.value);
            });
        });

        // Camera controls
        this.startCameraBtn.addEventListener('click', () => this.startCamera());
        this.stopCameraBtn.addEventListener('click', () => this.stopCamera());
        this.saveSnapshotBtn.addEventListener('click', () => this.saveSnapshot());

        // Upload controls
        this.browseButton.addEventListener('click', () => {
            this.imageUpload.click();
        });

        this.imageUpload.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleImageUpload(e.target.files[0]);
            }
        });

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.style.borderColor = 'var(--primary)';
            this.uploadArea.style.backgroundColor = 'var(--soft-stone)';
        });

        this.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadArea.style.borderColor = 'var(--hairline)';
            this.uploadArea.style.backgroundColor = 'var(--pale-green)';
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.style.borderColor = 'var(--hairline)';
            this.uploadArea.style.backgroundColor = 'var(--pale-green)';

            if (e.dataTransfer.files.length > 0) {
                this.handleImageUpload(e.dataTransfer.files[0]);
            }
        });

        // Settings
        this.confidenceSlider.addEventListener('input', (e) => {
            this.confidenceValue.textContent = e.target.value;
        });

        // Remove the problematic interval - video feed shows real-time results
        // Results are updated automatically by the server through the video pipeline
    }

    connectWebSocket() {
        // Socket.IO connection
        this.socket = io();

        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.modelsLoaded = true;
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
        });

        this.socket.on('results', (data) => {
            this.displayResults(data.results);
        });
    }

    switchMode(mode) {
        this.currentMode = mode;

        if (mode === 'camera') {
            this.uploadControls.style.display = 'none';
            document.getElementById('cameraControls').style.display = 'block';
        } else {
            this.stopCamera();
            document.getElementById('cameraControls').style.display = 'none';
            this.uploadControls.style.display = 'block';
        }
    }

    async startCamera() {
        try {
            this.showLoading('Starting camera...');

            // Explicitly tell the server to start the camera
            await fetch('/start_camera');

            // Start server-side video feed
            this.videoFeed.src = '/video_feed';

            // Show/hide elements
            this.videoFeed.style.display = 'block';
            this.videoPlaceholder.style.display = 'none';

            this.cameraActive = true;
            this.startCameraBtn.disabled = true;
            this.stopCameraBtn.disabled = false;
            this.saveSnapshotBtn.disabled = false;

            this.hideLoading();

        } catch (error) {
            console.error('Camera error:', error);
            alert('Failed to access camera: ' + error.message);
            this.hideLoading();
        }
    }

    stopCamera() {
        if (this.videoStream) {
            this.videoStream.getTracks().forEach(track => track.stop());
            this.videoStream = null;
        }

        this.cameraActive = false;
        this.videoFeed.style.display = 'none';
        this.videoPlaceholder.style.display = 'flex';

        this.startCameraBtn.disabled = false;
        this.stopCameraBtn.disabled = true;
        this.saveSnapshotBtn.disabled = true;

        // Clear results
        this.clearResults();
    }

    async startVideoStreaming() {
        // In a real implementation, this would send video frames to server
        // For now, we'll use the existing /video_feed endpoint
        this.videoFeed.src = '/video_feed';
    }

    async updateCameraResults() {
        try {
            const response = await fetch('/process_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    // Send current frame data here
                    timestamp: Date.now()
                })
            });

            if (response.ok) {
                const data = await response.json();
                if (data.results) {
                    this.displayResults(data.results);
                }
            }
        } catch (error) {
            console.error('Failed to update results:', error);
        }
    }

    async handleImageUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        this.showLoading('Processing image...');

        const formData = new FormData();
        formData.append('image', file);

        try {
            // Process image with increased timeout for large files
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000);

            const response = await fetch('/upload_image', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Image processing failed');
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Display processed image
            this.processedImage.src = 'data:image/jpeg;base64,' + data.frame;
            this.uploadedImageContainer.style.display = 'block';

            // Display results
            this.displayResults(data.results);

            this.hideLoading();

        } catch (error) {
            console.error('Upload error:', error);
            alert('Failed to process image: ' + error.message);
            this.hideLoading();
        }
    }

    displayResults(results) {
        if (!results || results.length === 0) {
            if (!this.clearResultsTimeout) {
                this.clearResultsTimeout = setTimeout(() => {
                    this.resultsContainer.innerHTML = `
                        <div class="empty-state">
                            <i class="fas fa-leaf fa-3x" style="margin-bottom: 20px; color: var(--hairline);"></i>
                            <p class="body-large">No leaves detected. Position a leaf clearly in view or try another image.</p>
                        </div>
                    `;
                    this.summaryStats.style.display = 'none';
                    this.clearResultsTimeout = null;
                }, 500); // 500ms debounce to prevent flickering
            }
            return;
        }

        // We have results, cancel any pending clear
        if (this.clearResultsTimeout) {
            clearTimeout(this.clearResultsTimeout);
            this.clearResultsTimeout = null;
        }

        this.lastResults = results;

        // Update summary statistics
        const healthyCount = results.filter(r => r.is_healthy).length;
        const diseasedCount = results.length - healthyCount;

        this.totalLeavesEl.textContent = results.length;
        this.healthyCountEl.textContent = healthyCount;
        this.diseasedCountEl.textContent = diseasedCount;

        this.summaryStats.style.display = 'grid';

        // Display individual results
        const resultsHTML = results.map(result => `
            <div class="result-card">
                <div class="result-header">
                    <span class="result-icon">
                        <i class="fas ${result.is_healthy ? 'fa-check-circle text-healthy' : 'fa-exclamation-triangle text-diseased'}"></i>
                    </span>
                    <span class="result-title mono-label">${result.plant}</span>
                    <span class="result-status ${result.is_healthy ? 'healthy' : 'diseased'} body-large">
                        ${result.disease.split(' - ')[1] || result.disease}
                    </span>
                </div>
                <div class="result-content">
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${result.confidence}%"></div>
                        <span class="confidence-text mono-label">${result.confidence}% confidence</span>
                    </div>
                </div>
                <div class="treatment-box research-table">
                    <p class="body">
                        <i class="fas fa-pills" style="margin-right: 8px;"></i>
                        <strong>Treatment:</strong> ${result.treatment}
                    </p>
                </div>
            </div>
        `).join('');

        this.resultsContainer.innerHTML = resultsHTML;
    }

    clearResults() {
        this.resultsContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-leaf"></i>
                <p>No detections yet. Start camera or upload an image to begin.</p>
            </div>
        `;
        this.summaryStats.style.display = 'none';
    }

    saveSnapshot() {
        if (this.currentMode === 'camera' && this.videoFeed.src) {
            // Create a temporary canvas to capture the current video frame
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = this.videoFeed.videoWidth;
            canvas.height = this.videoFeed.videoHeight;
            ctx.drawImage(this.videoFeed, 0, 0);

            // Download the image
            const link = document.createElement('a');
            link.download = `snapshot_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.jpg`;
            link.href = canvas.toDataURL('image/jpeg');
            link.click();

            alert('Snapshot saved successfully!');
        } else if (this.currentMode === 'upload' && this.processedImage.src) {
            // Download the processed uploaded image
            const link = document.createElement('a');
            link.download = `processed_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.jpg`;
            link.href = this.processedImage.src;
            link.click();

            alert('Snapshot saved successfully!');
        }
    }

    showLoading(message = 'Loading...') {
        this.loadingOverlay.querySelector('p').textContent = message;
        this.loadingOverlay.style.display = 'flex';
    }

    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.plantApp = new PlantDiseaseApp();
});