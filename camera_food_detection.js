// Food detection integration for camera feed
class FoodDetectionCamera {
    constructor(videoElementId = 'main-video', overlayCanvasId = 'detection-overlay') {
        this.apiEndpoint = 'http://localhost:5000';  // Fixed API endpoint
        this.isDetecting = false;
        this.detectionInterval = null;
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.videoElementId = videoElementId;
        this.overlayCanvas = document.getElementById(overlayCanvasId) || this.createOverlayCanvas();
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        this.foodCountElement = document.getElementById('food-count');
        this.detectionInfoElement = document.getElementById('detection-info');
        
        // Info overlay panel properties
        this.infoOverlay = null;
        this.infoInterval = null;
        this.createInfoOverlay();
    }

    createInfoOverlay() {
        // Create overlay container div
        this.infoOverlay = document.createElement('div');
        this.infoOverlay.id = 'info-overlay';
        this.infoOverlay.style.position = 'absolute';
        this.infoOverlay.style.top = '10px';
        this.infoOverlay.style.right = '10px';
        this.infoOverlay.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
        this.infoOverlay.style.color = 'white';
        this.infoOverlay.style.padding = '10px';
        this.infoOverlay.style.borderRadius = '5px';
        this.infoOverlay.style.fontSize = '14px';
        this.infoOverlay.style.zIndex = '20';
        this.infoOverlay.style.maxWidth = '300px';
        this.infoOverlay.style.fontFamily = 'Arial, sans-serif';
        
        // Add title
        const title = document.createElement('div');
        title.textContent = 'Demo Info Panel';
        title.style.fontWeight = 'bold';
        title.style.marginBottom = '8px';
        title.style.textAlign = 'center';
        this.infoOverlay.appendChild(title);
        
        // Add content div
        this.contentDiv = document.createElement('div');
        this.contentDiv.id = 'info-content';
        this.contentDiv.style.minHeight = '20px';
        this.infoOverlay.appendChild(this.contentDiv);
        
        // Insert after the video element
        const videoElement = document.getElementById(this.videoElementId);
        if (videoElement) {
            videoElement.parentNode.insertBefore(this.infoOverlay, videoElement.nextSibling);
        }
    }

    updateInfoDisplay() {
        // Array of random information to display
        const randomInfos = [
            `FPS: ${Math.floor(Math.random() * 30) + 1}`,
            `Objects: ${Math.floor(Math.random() * 10) + 1}`,
            `Confidence: ${Math.floor(Math.random() * 100) + 1}%`,
            `Resolution: ${Math.floor(Math.random() * 1000) + 720}p`,
            `Model: YOLOv8`,
            `Memory: ${(Math.random() * 8).toFixed(1)} GB`,
            `Temperature: ${Math.floor(Math.random() * 30) + 20}°C`,
            `Humidity: ${Math.floor(Math.random() * 50) + 30}%`,
            `Light: ${Math.floor(Math.random() * 100) + 1}%`,
            `Battery: ${Math.floor(Math.random() * 100) + 1}%`,
            `Network: ${['WiFi', 'Ethernet', 'LTE'][Math.floor(Math.random() * 3)]}`
        ];
        
        // Pick a random piece of information
        const randomInfo = randomInfos[Math.floor(Math.random() * randomInfos.length)];
        
        // Update the content
        this.contentDiv.textContent = randomInfo;
    }

    startInfoDisplay(intervalMs = 2000) {
        // Start updating the info display at regular intervals
        this.updateInfoDisplay(); // Show initial info
        this.infoInterval = setInterval(() => {
            this.updateInfoDisplay();
        }, intervalMs);
    }

    stopInfoDisplay() {
        if (this.infoInterval) {
            clearInterval(this.infoInterval);
            this.infoInterval = null;
        }
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
        const videoElement = document.getElementById(this.videoElementId);
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
                console.log("Detection successful:", result);
                this.drawDetections(result.detections);
                
                // Update food count display
                if (this.foodCountElement) {
                    this.foodCountElement.textContent = result.total_food_items || 0;
                }
                if (this.detectionInfoElement) {
                    if (result.total_food_items > 0) {
                        this.detectionInfoElement.style.display = 'block';
                    } else {
                        this.detectionInfoElement.style.display = 'none';
                    }
                }
                
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

            // Draw mask if available
            if (detection.mask) {
                console.log("mask:", detection.mask, "color: ", detection.class_color);
                this.drawMask(detection.mask, detection.class_color || 'rgba(255, 0, 0, 0.3)');
            } else {
                console.log("no mask");
            }

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

    drawMask(maskData, color = 'rgba(255, 0, 0, 0.3)') {
        // Create a temporary canvas to draw the mask
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        // Set canvas size to match the overlay
        tempCanvas.width = this.overlayCanvas.width;
        tempCanvas.height = this.overlayCanvas.height;
        
        // If maskData is an array of points (polygon), draw a polygon
        if (Array.isArray(maskData) && maskData.length > 0) {
            tempCtx.beginPath();
            tempCtx.moveTo(maskData[0][0], maskData[0][1]);
            
            for (let i = 1; i < maskData.length; i++) {
                tempCtx.lineTo(maskData[i][0], maskData[i][1]);
            }
            
            tempCtx.closePath();
            tempCtx.fillStyle = color;
            tempCtx.fill();
        } 
        // If maskData is a binary mask represented as an object with width/height/data
        else if (maskData && typeof maskData === 'object' && maskData.hasOwnProperty('data')) {
            // Create image data for the mask
            const imageData = new ImageData(
                new Uint8ClampedArray(maskData.data),
                maskData.width,
                maskData.height
            );
            
            // Draw the mask at the appropriate position
            tempCtx.putImageData(imageData, maskData.x || 0, maskData.y || 0);
            
            // Apply color to non-transparent pixels
            const outputImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
            const data = outputImageData.data;
            
            for (let i = 0; i < data.length; i += 4) {
                if (data[i + 3] > 0) { // If alpha channel is not zero
                    data[i] = parseInt(color.substring(1, 3), 16);     // R
                    data[i + 1] = parseInt(color.substring(3, 5), 16); // G
                    data[i + 2] = parseInt(color.substring(5, 7), 16); // B
                    data[i + 3] = Math.floor(255 * 0.3);               // A (30% opacity)
                }
            }
            
            tempCtx.putImageData(outputImageData, 0, 0);
        }
        
        // Draw the mask onto the main overlay canvas
        this.overlayCtx.drawImage(tempCanvas, 0, 0);
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
