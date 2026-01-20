// Food detection integration for camera feed
class FoodDetectionCamera {
    constructor(apiEndpoint = 'http://localhost:5000') {
        this.apiEndpoint = apiEndpoint;
        this.isDetecting = false;
        this.detectionInterval = null;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.overlayCanvas = document.getElementById('detection-overlay') || this.createOverlayCanvas();
        this.overlayCtx = this.overlayCanvas.getContext('2d');
    }

    createOverlayCanvas() {
        const overlay = document.createElement('canvas');
        overlay.id = 'detection-overlay';
        overlay.style.position = 'absolute';
        overlay.style.top = '0';
        overlay.style.left = '0';
        overlay.style.pointerEvents = 'none';
        overlay.style.zIndex = '10';
        
        // Insert after the video element
        const videoElement = document.getElementById('main-video');
        if (videoElement) {
            videoElement.parentNode.insertBefore(overlay, videoElement.nextSibling);
            this.setCanvasSize(videoElement);
        }
        
        return overlay;
    }

    setCanvasSize(videoElement) {
        this.overlayCanvas.width = videoElement.videoWidth;
        this.overlayCanvas.height = videoElement.videoHeight;
        this.overlayCanvas.style.width = videoElement.offsetWidth + 'px';
        this.overlayCanvas.style.height = videoElement.offsetHeight + 'px';
    }

    async detectFoodInFrame(videoElement) {
        // Resize the canvas to match video dimensions
        if (this.overlayCanvas.width !== videoElement.videoWidth || 
            this.overlayCanvas.height !== videoElement.videoHeight) {
            this.setCanvasSize(videoElement);
        }

        // Draw current video frame to canvas
        this.canvas.width = videoElement.videoWidth;
        this.canvas.height = videoElement.videoHeight;
        this.ctx.drawImage(videoElement, 0, 0, this.canvas.width, this.canvas.height);

        // Convert canvas to base64 image
        const imageData = this.canvas.toDataURL('image/jpeg', 0.8).split(',')[1];

        try {
            // Send to backend for food detection
            const response = await fetch(`${this.apiEndpoint}/detect_food`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image: imageData })
            });

            const result = await response.json();

            if (result.success) {
                this.drawDetections(result.detections);
                return result;
            } else {
                console.error('Detection failed:', result.error);
                return null;
            }
        } catch (error) {
            console.error('Error during food detection:', error);
            return null;
        }
    }

    drawDetections(detections) {
        // Clear previous drawings
        this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

        detections.forEach(detection => {
            const box = detection.box;
            const confidence = Math.round(detection.score * 100);

            // Draw bounding box
            this.overlayCtx.strokeStyle = '#FF0000';
            this.overlayCtx.lineWidth = 2;
            this.overlayCtx.strokeRect(
                box.xmin,
                box.ymin,
                box.xmax - box.xmin,
                box.ymax - box.ymin
            );

            // Draw label background
            this.overlayCtx.fillStyle = 'rgba(255, 0, 0, 0.7)';
            this.overlayCtx.fillRect(
                box.xmin,
                box.ymin - 20,
                detection.label.length * 10 + 30,
                20
            );

            // Draw label text
            this.overlayCtx.fillStyle = 'white';
            this.overlayCtx.font = '14px Arial';
            this.overlayCtx.fillText(
                `${detection.label} (${confidence}%)`,
                box.xmin + 5,
                box.ymin - 5
            );
        });
    }

    startDetection(videoElement, intervalMs = 1000) {
        if (this.isDetecting) {
            console.log('Detection already running');
            return;
        }

        this.isDetecting = true;
        console.log('Starting food detection...');

        const detect = async () => {
            if (this.isDetecting && videoElement && !videoElement.paused && !videoElement.ended) {
                await this.detectFoodInFrame(videoElement);
            }
        };

        // Run detection immediately and then at intervals
        detect();
        this.detectionInterval = setInterval(detect, intervalMs);
    }

    stopDetection() {
        this.isDetecting = false;
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        
        // Clear the overlay
        if (this.overlayCtx) {
            this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        }
        
        console.log('Stopped food detection');
    }

    async testConnection() {
        try {
            const response = await fetch(`${this.apiEndpoint}/health`);
            const health = await response.json();
            return health.status === 'healthy';
        } catch (error) {
            console.error('Cannot connect to food detection API:', error);
            return false;
        }
    }
}
