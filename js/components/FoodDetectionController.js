/**
 * Food Detection Controller Component
 * Handles food detection from video and image files using backend API
 */

import { createComponent, showElement, hideElement, updateButton, updateStatus, createProgressBar, showNotification } from '../lib/componentUtils.js';

export class FoodDetectionController {
    constructor() {
        this.detectionResults = [];
        this.currentVideoFile = null;
        this.detectionInProgress = false;
        this.progressBar = null;
        this.isDetecting = false;
        this.detectionInterval = null;
        this.detectionOverlay = null;
        this.lastDetectionResults = [];
        this.ctx = null;
    }

    /**
     * Initialize food detection controller
     */
    initialize() {
        this.setupEventListeners();
        this.progressBar = createProgressBar('progress-container', 'detection-progress', 'progress-text');
        this.progressBar.hide();
        
        // Get UI elements for real-time detection
        this.detectionButton = document.getElementById('food-detection-btn');
        this.detectionStatus = document.getElementById('detection-status');
        this.detectionOverlay = document.getElementById('detection-overlay');
        this.mainVideo = document.getElementById('main-video');
        
        // Initialize canvas context
        if (this.detectionOverlay) {
            this.ctx = this.detectionOverlay.getContext('2d');
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        document.addEventListener("toggleFoodDetection", () => {
            this.toggleFoodDetection();
        });

        // Clear results button
        const clearResultsBtn = document.getElementById('clear-results-btn');
        if (clearResultsBtn) {
            clearResultsBtn.addEventListener('click', () => this.clearResults());
        }
    }

    /**
     * Toggle food detection on/off
     */
    toggleFoodDetection() {
        if (this.isDetecting) {
            this.stopFoodDetection();
        } else {
            this.startFoodDetection();
        }
    }

    /**
     * Start food detection
     */
    async startFoodDetection() {
        if (!this.mainVideo || !this.mainVideo.srcObject) {
            updateStatus('detection-status', 'Camera not active', 'error');
            return;
        }

        this.isDetecting = true;
        updateButton(this.detectionButton, 'Stop Food Detection', false);
        updateStatus('detection-status', 'Food Detection: Active', 'success');

        // Start detection interval
        this.detectionInterval = setInterval(async () => {
            await this.performDetection();
        }, 2000); // Detect every 2 seconds

        // Perform initial detection
        await this.performDetection();
    }

    /**
     * Stop food detection
     */
    stopFoodDetection() {
        this.isDetecting = false;
        
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }

        updateButton(this.detectionButton, 'Start Food Detection', false);
        updateStatus('detection-status', 'Food Detection: Off', 'info');
        
        // Clear detection overlay
        this.clearDetectionOverlay();
    }

    /**
     * Perform food detection by capturing current frame and sending to API
     */
    async performDetection() {
        if (!this.mainVideo || !this.mainVideo.srcObject || !this.ctx) {
            return;
        }

        try {
            // Create a temporary canvas to capture the current frame
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = this.mainVideo.videoWidth;
            tempCanvas.height = this.mainVideo.videoHeight;
            const tempCtx = tempCanvas.getContext('2d');

            // Draw the current video frame to the temporary canvas
            tempCtx.drawImage(this.mainVideo, 0, 0, tempCanvas.width, tempCanvas.height);

            console.log("canvas width:", tempCanvas.width, "canvas height:", tempCanvas.height);

            // Convert to base64 (remove the data:image/jpeg;base64, prefix)
            const imageData = tempCanvas.toDataURL('image/jpeg');
            const base64Image = imageData.split(',')[1];
            
            // Send to backend API
            const response = await fetch('/detect_food', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Image
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            // Check if result has the expected structure
            if (result.success && result.detections) {
                this.lastDetectionResults = result.detections;
                this.drawDetections(this.lastDetectionResults);
                this.displayDetections(result); // Display results in the UI
                updateStatus('detection-status', `Detected ${this.lastDetectionResults.length} food items`, 'success');
            } else {
                throw new Error(result.error || 'Detection failed');
            }
        } catch (error) {
            console.error('Error during food detection:', error);
            updateStatus('detection-status', `Detection error: ${error.message}`, 'error');
        }
    }

    /**
     * Draw detection results on the overlay canvas
     */
    drawDetections(detections) {
        if (!this.ctx || !this.detectionOverlay || !this.mainVideo) {
            return;
        }

        // Get the actual display size of the video element
        const videoRect = this.mainVideo.getBoundingClientRect();
        const videoDisplayWidth = videoRect.width;
        const videoDisplayHeight = videoRect.height;

        // Set canvas size to match video display size
        this.detectionOverlay.style.width = `${videoDisplayWidth}px`;
        this.detectionOverlay.style.height = `${videoDisplayHeight}px`;
        this.detectionOverlay.width = videoDisplayWidth;
        this.detectionOverlay.height = videoDisplayHeight;

        // Clear previous drawings
        this.ctx.clearRect(0, 0, this.detectionOverlay.width, this.detectionOverlay.height);

        // Calculate scale factors from video resolution to display size
        const scaleX = videoDisplayWidth / this.mainVideo.videoWidth;
        const scaleY = videoDisplayHeight / this.mainVideo.videoHeight;

        // Update detection info
        const detectionInfo = document.getElementById('detection-info');
        const foodCount = document.getElementById('food-count');
        
        if (detections.length > 0) {
            if (detectionInfo) {
                detectionInfo.classList.remove('hidden');
            }
            if (foodCount) {
                foodCount.textContent = detections.length;
            }
        } else {
            if (detectionInfo) {
                detectionInfo.classList.add('hidden');
            }
        }

        // Draw each detection
        detections.forEach(detection => {
            const { label, score, box, mask } = detection;
            if (label == 'person') {
                return;
            }
            
            // Scale bounding box coordinates to match display size
            const scaledX1 = box.xmin * scaleX;
            const scaledY1 = box.ymin * scaleY;
            const scaledWidth = (box.xmax - box.xmin) * scaleX;
            const scaledHeight = (box.ymax - box.ymin) * scaleY;
            
            // Draw segmentation mask if available
            if (mask && mask.length > 0) {
                // Create a temporary canvas for the mask
                const maskCanvas = document.createElement('canvas');
                const maskCtx = maskCanvas.getContext('2d');
                
                // Calculate mask dimensions
                const maskHeight = mask.length;
                const maskWidth = mask[0].length;
                
                // Set mask canvas size to match the original mask dimensions
                maskCanvas.width = maskWidth;
                maskCanvas.height = maskHeight;
                
                // Create image data from mask array
                const maskImageData = maskCtx.createImageData(maskCanvas.width, maskCanvas.height);
                const maskData = maskImageData.data;
                
                // Fill the mask data
                for (let y = 0; y < maskHeight; y++) {
                    for (let x = 0; x < maskWidth; x++) {
                        const maskValue = mask[y] && mask[y][x] ? mask[y][x] : 0;
                        const pixelIndex = (y * maskWidth + x) * 4;
                        
                        if (maskValue > 0) {
                            // Semi-transparent green for mask
                            maskData[pixelIndex] = 0;     // R
                            maskData[pixelIndex + 1] = 255; // G
                            maskData[pixelIndex + 2] = 0;   // B
                            maskData[pixelIndex + 3] = 100; // A (transparency)
                        } else {
                            // Transparent for non-mask areas
                            maskData[pixelIndex] = 0;
                            maskData[pixelIndex + 1] = 0;
                            maskData[pixelIndex + 2] = 0;
                            maskData[pixelIndex + 3] = 0;
                        }
                    }
                }
                
                // Put the mask image data on the mask canvas
                maskCtx.putImageData(maskImageData, 0, 0);
                
                // Calculate the aspect ratio of the mask
                const maskAspectRatio = maskWidth / maskHeight;
                const boundingBoxAspectRatio = scaledWidth / scaledHeight;
                
                // Calculate the dimensions and position to maintain aspect ratio
                let drawWidth, drawHeight, drawX, drawY;
                
                if (maskAspectRatio > boundingBoxAspectRatio) {
                    // Mask is wider than the bounding box, fit to width
                    drawWidth = scaledWidth;
                    drawHeight = scaledWidth / maskAspectRatio;
                    drawX = scaledX1;
                    drawY = scaledY1 + (scaledHeight - drawHeight) / 2;
                } else {
                    // Mask is taller than the bounding box, fit to height
                    drawHeight = scaledHeight;
                    drawWidth = scaledHeight * maskAspectRatio;
                    drawX = scaledX1 + (scaledWidth - drawWidth) / 2;
                    drawY = scaledY1;
                }
                
                // Draw the mask on the main canvas with proper aspect ratio
                this.ctx.drawImage(maskCanvas, drawX, drawY, drawWidth, drawHeight);
            }
            
            // Draw bounding box
            this.ctx.strokeStyle = '#00FF00';
            this.ctx.lineWidth = 2;
            this.ctx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);
            
            // Draw label background
            const text = `${label} (${Math.round(score * 100)}%)`;
            this.ctx.font = '16px Arial';
            const textWidth = this.ctx.measureText(text).width;
            this.ctx.fillStyle = 'rgba(0, 255, 0, 0.7)';
            this.ctx.fillRect(scaledX1, scaledY1 - 20, textWidth + 4, 20);
            
            // Save the current context state
            this.ctx.save();
            
            // Flip the context horizontally for text
            this.ctx.scale(-1, 1);
            
            // Draw label text (flipped position to account for the scale transformation)
            this.ctx.fillStyle = '#000000';
            this.ctx.fillText(text, -(scaledX1 + 2), scaledY1 - 5);
            
            // Restore the context state
            this.ctx.restore();
        });
    }

    /**
     * Clear the detection overlay
     */
    clearDetectionOverlay() {
        if (this.ctx && this.detectionOverlay) {
            this.ctx.clearRect(0, 0, this.detectionOverlay.width, this.detectionOverlay.height);
        }
        
        // Hide detection info
        const detectionInfo = document.getElementById('detection-info');
        if (detectionInfo) {
            detectionInfo.classList.add('hidden');
        }
        
        // Reset last detection results
        this.lastDetectionResults = [];
    }


    /**
     * Display detection results
     * @param {Object} results - Detection results from API
     */
    displayDetections(results) {
        this.detectionResults = results.detections || [];
        
        const resultsContainer = document.getElementById('detection-results');
        if (!resultsContainer) return;

        // Clear previous results
        resultsContainer.innerHTML = '';

        if (this.detectionResults.length === 0) {
            resultsContainer.innerHTML = '<p class="text-gray-400 text-center py-4">No food items detected</p>';
            return;
        }

        // Create results header
        const header = document.createElement('div');
        header.className = 'flex justify-between items-center mb-4';
        header.innerHTML = `
            <h3 class="text-lg font-semibold">Detection Results</h3>
            <span class="text-sm text-gray-400">${this.detectionResults.length} items found</span>
        `;
        resultsContainer.appendChild(header);

        // Create results list
        const list = document.createElement('div');
        list.className = 'space-y-2';

        this.detectionResults.forEach((detection, index) => {
            const item = createComponent('detection-item');
            
            // Update item content
            const nameElement = item.querySelector('.detection-name');
            const confidenceElement = item.querySelector('.detection-confidence');
            const caloriesElement = item.querySelector('.detection-calories');
            
            if (nameElement) nameElement.textContent = detection.label || 'Unknown';
            if (confidenceElement) confidenceElement.textContent = `${Math.round((detection.score || 0) * 100)}%`;
            if (caloriesElement) caloriesElement.textContent = detection.calories || 'N/A';
            
            // Add click event to show details
            const detailsBtn = item.querySelector('.detection-details-btn');
            if (detailsBtn) {
                detailsBtn.addEventListener('click', () => this.showDetectionDetails(index));
            }
            
            list.appendChild(item);
        });

        resultsContainer.appendChild(list);
        showElement(resultsContainer);

        // Show clear button
        const clearBtn = document.getElementById('clear-results-btn');
        if (clearBtn) {
            showElement(clearBtn);
        }
    }

    /**
     * Show detection details
     * @param {number} index - Index of detection item
     */
    showDetectionDetails(index) {
        if (index < 0 || index >= this.detectionResults.length) return;
        
        const detection = this.detectionResults[index];
        
        // Create modal
        const modal = createComponent('detection-modal');
        
        // Update modal content
        const modalTitle = modal.querySelector('.modal-title');
        const modalBody = modal.querySelector('.modal-body');
        
        if (modalTitle) modalTitle.textContent = detection.label || 'Unknown Food';
        
        if (modalBody) {
            modalBody.innerHTML = `
                <div class="space-y-4">
                    <div>
                        <h4 class="font-semibold text-sm text-gray-400">Confidence</h4>
                        <p>${Math.round((detection.score || 0) * 100)}%</p>
                    </div>
                    <div>
                        <h4 class="font-semibold text-sm text-gray-400">Bounding Box</h4>
                        <p>XMin: ${detection.box?.xmin || 'N/A'}, YMin: ${detection.box?.ymin || 'N/A'}</p>
                        <p>XMax: ${detection.box?.xmax || 'N/A'}, YMax: ${detection.box?.ymax || 'N/A'}</p>
                    </div>
                    ${detection.mask ? `
                    <div>
                        <h4 class="font-semibold text-sm text-gray-400">Segmentation Mask</h4>
                        <p>Available (${detection.mask.length} rows)</p>
                    </div>
                    ` : ''}
                </div>
            `;
        }
        
        // Add to body
        document.body.appendChild(modal);
        
        // Setup close button
        const closeBtn = modal.querySelector('.modal-close-btn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                if (modal.parentNode) {
                    modal.parentNode.removeChild(modal);
                }
            });
        }
        
        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                if (modal.parentNode) {
                    modal.parentNode.removeChild(modal);
                }
            }
        });
    }

    /**
     * Clear detection results
     */
    clearResults() {
        this.detectionResults = [];
        
        const resultsContainer = document.getElementById('detection-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = '';
            hideElement(resultsContainer);
        }
        
        const clearBtn = document.getElementById('clear-results-btn');
        if (clearBtn) {
            hideElement(clearBtn);
        }
        
        updateStatus('main-status', 'Results cleared');
    }

}
