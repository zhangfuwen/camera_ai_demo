/**
 * Mosaic Effect Controller Component
 * Handles applying mosaic/pixelation effect to video streams
 */

import { showElement, hideElement, updateStatus } from '../lib/componentUtils.js';

export class MosaicEffectController {
    constructor() {
        this.videoElement = null;
        this.canvas = null;
        this.ctx = null;
        this.isProcessing = false;
        this.pixelSize = 10; // Default pixelation size
        this.isEnabled = false;
        this.animationFrameId = null;
        this.rotationAngle = 0;
        this.isMirrored = false;
        this.isRedFilled = false;
    }

    /**
     * Initialize mosaic effect controller
     * @param {HTMLVideoElement} videoElement - Video element to apply effect to
     * @param {CameraController} cameraController - Reference to camera controller for transformations
     */
    initialize(videoElement, cameraController = null) {
        this.videoElement = videoElement;
        this.cameraController = cameraController;
        
        // Create canvas for processing
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'mosaic-canvas'; // Add ID for easy identification
        this.ctx = this.canvas.getContext('2d');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Hide the original video and show the canvas
        this.setupCanvasDisplay();
        
        // Initialize canvas dimensions to match video actual resolution
        // Defer this operation until the video element is properly rendered
        setTimeout(() => {
            if (this.videoElement) {
                // Prioritize actual video dimensions over display dimensions
                this.canvas.width = this.videoElement.videoWidth || this.videoElement.offsetWidth || 640;
                this.canvas.height = this.videoElement.videoHeight || this.videoElement.offsetHeight || 480;
            }
        }, 0);
    }

    /**
     * Setup event listeners for controls
     */
    setupEventListeners() {
        // Pixel size slider
        const pixelSizeSlider = document.getElementById('pixel-size-slider');
        if (pixelSizeSlider) {
            pixelSizeSlider.addEventListener('input', (e) => {
                this.pixelSize = parseInt(e.target.value);
                const pixelSizeValue = document.getElementById('pixel-size-value');
                if (pixelSizeValue) {
                    pixelSizeValue.textContent = `${this.pixelSize}px`;
                }
            });
        }

        // Toggle mosaic effect button
        const toggleMosaicBtn = document.getElementById('toggle-mosaic-btn');
        if (toggleMosaicBtn) {
            toggleMosaicBtn.addEventListener('click', () => {
                this.toggleMosaicEffect();
            });
        }
        
        // Toggle red canvas button
        const toggleRedCanvasBtn = document.getElementById('toggle-red-canvas-btn');
        if (toggleRedCanvasBtn) {
            toggleRedCanvasBtn.addEventListener('click', () => {
                this.toggleRedCanvas();
            });
        }
        
        // Add event listener for the save canvas button
        const saveCanvasBtn = document.getElementById('save-canvas-btn');
        if (saveCanvasBtn) {
            saveCanvasBtn.addEventListener('click', () => {
                this.saveCanvasAsJPEG();
            });
        }
    }

    /**
     * Setup canvas display
     */
    setupCanvasDisplay() {
        if (!this.videoElement || !this.canvas) return;
        
        // Position the canvas exactly where the video is
        const videoContainer = this.videoElement.parentElement;
        
        // Set canvas styles to match the video
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = (this.videoElement.offsetTop || 0) + 'px';
        this.canvas.style.left = (this.videoElement.offsetLeft || 0) + 'px';
        this.canvas.style.width = (this.videoElement.offsetWidth || this.videoElement.videoWidth || 640) + 'px';
        this.canvas.style.height = (this.videoElement.offsetHeight || this.videoElement.videoHeight || 480) + 'px';
        this.canvas.style.objectFit = 'contain';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.zIndex = '9'; // Below detection overlay (z-index 10)
        
        // Initially show the canvas but make it transparent so it doesn't interfere with video display
        this.canvas.style.display = 'block';
        this.canvas.style.opacity = '0';
        
        // Add canvas to the container
        videoContainer.appendChild(this.canvas);
        
        // Add a resize observer to sync canvas size with video element
        if (window.ResizeObserver) {
            this.resizeObserver = new ResizeObserver(entries => {
                for (let entry of entries) {
                    if (entry.target === this.videoElement) {
                        this.updateCanvasPosition();
                    }
                }
            });
            this.resizeObserver.observe(this.videoElement, { box: 'border-box' });
        }
    }
    
    /**
     * Update canvas position and size to match video element
     */
    updateCanvasPosition() {
        if (!this.videoElement || !this.canvas) return;
        
        this.canvas.style.top = (this.videoElement.offsetTop || 0) + 'px';
        this.canvas.style.left = (this.videoElement.offsetLeft || 0) + 'px';
        this.canvas.style.width = (this.videoElement.offsetWidth || this.videoElement.videoWidth || 640) + 'px';
        this.canvas.style.height = (this.videoElement.offsetHeight || this.videoElement.videoHeight || 480) + 'px';
        
        // Update canvas internal dimensions to match display size, not the video resolution
        // This ensures the canvas matches the visible video area with object-contain sizing
        this.canvas.width = this.videoElement.offsetWidth || this.videoElement.videoWidth || 640;
        this.canvas.height = this.videoElement.offsetHeight || this.videoElement.videoHeight || 480;
    }

    /**
     * Toggle mosaic effect on/off
     */
    toggleMosaicEffect() {
        this.isEnabled = !this.isEnabled;
        
        // Update button state
        const mosaicButton = document.getElementById('mosaic-button');
        if (mosaicButton) {
            mosaicButton.classList.toggle('bg-blue-500', this.isEnabled);
            mosaicButton.classList.toggle('bg-gray-500', !this.isEnabled);
            mosaicButton.textContent = this.isEnabled ? 'Disable Mosaic' : 'Enable Mosaic';
        }
        
        // Show/hide mosaic controls
        const mosaicControls = document.getElementById('mosaic-controls');
        if (mosaicControls) {
            mosaicControls.style.display = this.isEnabled ? 'block' : 'none';
        }
        
        if (this.isEnabled) {
            this.startProcessing();
        } else {
            this.stopProcessing();
        }
    }
    
    /**
     * Toggle between red fill and mosaic effect
     */
    toggleRedCanvas() {
        if (!this.canvas || !this.ctx) return;
        
        // Toggle red fill state
        if (!this.isRedFilled) {
            // Fill canvas with red
            this.ctx.fillStyle = 'red';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            this.isRedFilled = true;
            
            // Update button text
            const redCanvasBtn = document.getElementById('toggle-red-canvas-btn');
            if (redCanvasBtn) {
                redCanvasBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    Show Mosaic
                `;
            }
        } else {
            // Restore mosaic effect if enabled
            this.isRedFilled = false;
            
            // Update button text
            const redCanvasBtn = document.getElementById('toggle-red-canvas-btn');
            if (redCanvasBtn) {
                redCanvasBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    Fill Canvas Red
                `;
            }
            
            // If mosaic is enabled, restart processing to show the mosaic effect
            if (this.isEnabled) {
                this.stopProcessing();
                this.startProcessing();
            } else {
                // Clear the canvas if mosaic is not enabled
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            }
        }
    }

    /**
     * Start processing video with mosaic effect
     */
    startProcessing() {
        if (!this.videoElement || !this.canvas || !this.ctx) return;
        
        this.isProcessing = true;
        
        // Show canvas by making it opaque and hide video
        if (this.canvas) {
            this.canvas.style.opacity = '1';
        }
        
        // Update canvas dimensions to match current video display size
        this.updateCanvasPosition();
        
        if (this.videoElement) {
            this.videoElement.style.opacity = '0';
        }
        
        // Wait for video to be loaded before starting processing
        const waitForVideo = () => {
            if (this.videoElement.readyState >= 2) { // HAVE_CURRENT_DATA or greater
                // Start processing loop
                this.processFrame();
            } else {
                setTimeout(waitForVideo, 100);
            }
        };
        
        waitForVideo();
        
        // Test: Temporarily fill canvas with red to verify it's working
        // Uncomment the next lines to see a red canvas
        /*
        if (this.canvas && this.ctx) {
            const canvasWidth = this.canvas.width;
            const canvasHeight = this.canvas.height;
            this.ctx.fillStyle = 'red';
            this.ctx.fillRect(0, 0, canvasWidth, canvasHeight);
        }
        */
    }

    /**
     * Stop processing video
     */
    stopProcessing() {
        this.isProcessing = false;
        
        // Disconnect the resize observer to prevent memory leaks
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        
        // Make canvas transparent to show video again
        if (this.canvas) {
            this.canvas.style.opacity = '0';
        }
        if (this.videoElement) {
            this.videoElement.style.opacity = '1';
        }
        
        // Cancel animation frame
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    /**
     * Process a single frame with mosaic effect
     */
    processFrame() {
        if (!this.isProcessing || !this.videoElement || this.videoElement.readyState !== 4) return;
        
        // Sync rotation and mirroring from camera controller if available
        if (this.cameraController) {
            this.rotationAngle = this.cameraController.rotationAngle;
            this.isMirrored = this.cameraController.isMirrored;
        }
        
        // Update canvas dimensions to match current video display size
        // This ensures the canvas internal dimensions match the display dimensions
        this.updateCanvasPosition();
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // If red fill is active, fill with red and return early
        if (this.isRedFilled) {
            this.ctx.fillStyle = 'red';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // Continue processing
            this.animationFrameId = requestAnimationFrame(() => this.processFrame());
            return;
        }
        
        // Calculate the aspect ratio to properly draw the video content
        // The video element uses object-contain which maintains aspect ratio
        const videoNaturalAspectRatio = this.videoElement.videoWidth / this.videoElement.videoHeight;
        const canvasAspectRatio = this.canvas.width / this.canvas.height;
        
        let drawWidth, drawHeight, offsetX, offsetY;
        
        // Determine how the video would be scaled in the canvas to maintain aspect ratio
        if (canvasAspectRatio > videoNaturalAspectRatio) {
            // Canvas is wider relative to video's aspect ratio - letterbox effect
            drawHeight = this.canvas.height;
            drawWidth = drawHeight * videoNaturalAspectRatio;
            offsetX = (this.canvas.width - drawWidth) / 2;
            offsetY = 0;
        } else {
            // Canvas is taller relative to video's aspect ratio - pillarbox effect
            drawWidth = this.canvas.width;
            drawHeight = drawWidth / videoNaturalAspectRatio;
            offsetX = 0;
            offsetY = (this.canvas.height - drawHeight) / 2;
        }
        
        // Save the current context state
        this.ctx.save();
        
        // Apply transformations (mirror and rotation)
        // First translate to the center of where the video should be drawn
        this.ctx.translate(offsetX + drawWidth / 2, offsetY + drawHeight / 2);
        
        // Apply mirror (scaleX(-1))
        if (this.isMirrored) {
            this.ctx.scale(-1, 1);
        }
        
        // Apply rotation
        this.ctx.rotate(this.rotationAngle * Math.PI / 180); // Convert degrees to radians
        
        // Draw the video frame to canvas with correct aspect ratio
        // Draw at (-drawWidth/2, -drawHeight/2) relative to the translated center
        this.ctx.drawImage(
            this.videoElement,
            -drawWidth / 2,
            -drawHeight / 2,
            drawWidth,
            drawHeight
        );
        
        // Restore the context state
        this.ctx.restore();
        
        // Apply mosaic effect to the transformed image
        this.applyMosaicEffect();
        
        // Continue processing
        this.animationFrameId = requestAnimationFrame(() => this.processFrame());
    }

    /**
     * Apply mosaic/pixelation effect to the canvas
     */
    applyMosaicEffect() {
        if (!this.ctx || !this.canvas) return;
        
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Get image data
        const imageData = this.ctx.getImageData(0, 0, width, height);
        const data = imageData.data;
        
        // Apply pixelation
        for (let y = 0; y < height; y += this.pixelSize) {
            for (let x = 0; x < width; x += this.pixelSize) {
                // Calculate the average color for this pixel block
                let r = 0, g = 0, b = 0, a = 0;
                let count = 0;
                
                for (let dy = 0; dy < this.pixelSize && y + dy < height; dy++) {
                    for (let dx = 0; dx < this.pixelSize && x + dx < width; dx++) {
                        const index = ((y + dy) * width + (x + dx)) * 4;
                        r += data[index];
                        g += data[index + 1];
                        b += data[index + 2];
                        a += data[index + 3];
                        count++;
                    }
                }
                
                // Calculate average
                r = Math.floor(r / count);
                g = Math.floor(g / count);
                b = Math.floor(b / count);
                a = Math.floor(a / count);
                
                // Apply average color to all pixels in this block
                for (let dy = 0; dy < this.pixelSize && y + dy < height; dy++) {
                    for (let dx = 0; dx < this.pixelSize && x + dx < width; dx++) {
                        const index = ((y + dy) * width + (x + dx)) * 4;
                        data[index] = r;
                        data[index + 1] = g;
                        data[index + 2] = b;
                        data[index + 3] = a;
                    }
                }
            }
        }
        
        // Put the modified image data back
        this.ctx.putImageData(imageData, 0, 0);
    }

    /**
     * Update pixel size
     * @param {number} newSize - New pixel size
     */
    setPixelSize(newSize) {
        this.pixelSize = Math.max(1, newSize); // Ensure minimum size is 1
    }
    
    /**
     * Save the canvas content as a JPEG image
     */
    saveCanvasAsJPEG() {
        if (!this.canvas) {
            console.error('Canvas not found');
            return;
        }
        
        try {
            // Temporarily make canvas visible if it's transparent to ensure content is rendered
            const originalOpacity = this.canvas.style.opacity;
            const originalDisplay = this.canvas.style.display;
            
            // Make canvas visible for rendering
            this.canvas.style.display = 'block';
            this.canvas.style.opacity = '1';
            
            // Ensure canvas has current content by forcing a render if needed
            if (this.videoElement && this.ctx) {
                // Update canvas dimensions to match video's actual resolution if available
                if (this.videoElement.videoWidth && this.videoElement.videoHeight) {
                    // Use the actual video resolution to maintain correct aspect ratio
                    this.canvas.width = this.videoElement.videoWidth;
                    this.canvas.height = this.videoElement.videoHeight;
                } else {
                    // Fallback to display size if video metadata isn't ready yet
                    this.updateCanvasPosition();
                }
                
                // Clear canvas
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                
                // Draw video frame based on current state
                if (this.isRedFilled) {
                    // Draw red fill if red fill is active
                    this.ctx.fillStyle = 'red';
                    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
                } else if (this.isEnabled) {
                    // If mosaic is enabled, draw the video frame and apply mosaic effect
                    // Calculate the aspect ratio to properly draw the video content
                    // The video element uses object-contain which maintains aspect ratio
                    const videoNaturalAspectRatio = this.videoElement.videoWidth / this.videoElement.videoHeight;
                    const canvasAspectRatio = this.canvas.width / this.canvas.height;
                    
                    let drawWidth, drawHeight, offsetX, offsetY;
                    
                    // Determine how the video would be scaled in the canvas to maintain aspect ratio
                    if (canvasAspectRatio > videoNaturalAspectRatio) {
                        // Canvas is wider relative to video's aspect ratio - letterbox effect
                        drawHeight = this.canvas.height;
                        drawWidth = drawHeight * videoNaturalAspectRatio;
                        offsetX = (this.canvas.width - drawWidth) / 2;
                        offsetY = 0;
                    } else {
                        // Canvas is taller relative to video's aspect ratio - pillarbox effect
                        drawWidth = this.canvas.width;
                        drawHeight = drawWidth / videoNaturalAspectRatio;
                        offsetX = 0;
                        offsetY = (this.canvas.height - drawHeight) / 2;
                    }
                    
                    // Save the current context state
                    this.ctx.save();
                    
                    // Apply transformations (mirror and rotation)
                    // First translate to the center of where the video should be drawn
                    this.ctx.translate(offsetX + drawWidth / 2, offsetY + drawHeight / 2);
                    
                    // Apply mirror (scaleX(-1))
                    if (this.isMirrored) {
                        this.ctx.scale(-1, 1);
                    }
                    
                    // Apply rotation
                    this.ctx.rotate(this.rotationAngle * Math.PI / 180); // Convert degrees to radians
                    
                    // Draw the video frame to canvas with correct aspect ratio
                    // Draw at (-drawWidth/2, -drawHeight/2) relative to the translated center
                    this.ctx.drawImage(
                        this.videoElement,
                        -drawWidth / 2,
                        -drawHeight / 2,
                        drawWidth,
                        drawHeight
                    );
                    
                    // Restore the context state
                    this.ctx.restore();
                    
                    // Apply mosaic effect to the transformed image
                    this.applyMosaicEffect();
                } else {
                    // Otherwise, draw the raw video frame
                    this.ctx.drawImage(this.videoElement, 0, 0, this.canvas.width, this.canvas.height);
                }
            }
            
            // Create a temporary canvas to ensure proper JPEG encoding
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            
            tempCanvas.width = this.canvas.width;
            tempCanvas.height = this.canvas.height;
            
            // Draw the current canvas content to the temporary canvas
            tempCtx.drawImage(this.canvas, 0, 0);
            
            // Convert to JPEG blob
            tempCanvas.toBlob((blob) => {
                if (blob) {
                    // Create download link
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `mosaic-effect-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.jpg`;
                    
                    // Trigger download
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    
                    // Clean up
                    URL.revokeObjectURL(url);
                    
                    console.log('Canvas saved as JPEG successfully');
                    
                    // Log the dimensions of the saved image
                    console.log(`Saved image dimensions: ${tempCanvas.width}x${tempCanvas.height}`);
                } else {
                    console.error('Failed to create JPEG blob');
                }
                
                // Restore original visibility
                this.canvas.style.opacity = originalOpacity;
                this.canvas.style.display = originalDisplay;
            }, 'image/jpeg', 0.9); // 90% quality JPEG
        } catch (error) {
            console.error('Error saving canvas as JPEG:', error);
        }
    }
}
