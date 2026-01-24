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
    }

    /**
     * Initialize mosaic effect controller
     * @param {HTMLVideoElement} videoElement - Video element to apply effect to
     */
    initialize(videoElement) {
        this.videoElement = videoElement;
        
        // Create canvas for processing
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Hide the original video and show the canvas
        this.setupCanvasDisplay();
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
    }

    /**
     * Setup canvas display
     */
    setupCanvasDisplay() {
        if (!this.videoElement || !this.canvas) return;
        
        // Position the canvas exactly where the video is
        const videoRect = this.videoElement.getBoundingClientRect();
        const videoContainer = this.videoElement.parentElement;
        
        // Set canvas styles to match the video
        this.canvas.style.position = 'absolute';
        this.canvas.style.top = '0';
        this.canvas.style.left = '0';
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.canvas.style.objectFit = 'contain';
        this.canvas.style.pointerEvents = 'none';
        this.canvas.style.zIndex = '5'; // Below detection overlay (z-index 10)
        
        // Initially hide the canvas
        this.canvas.style.display = 'none';
        
        // Add canvas to the container
        videoContainer.appendChild(this.canvas);
    }

    /**
     * Toggle mosaic effect on/off
     */
    toggleMosaicEffect() {
        this.isEnabled = !this.isEnabled;
        
        const toggleBtn = document.getElementById('toggle-mosaic-btn');
        const mosaicControls = document.getElementById('mosaic-controls');
        
        if (this.isEnabled) {
            // Enable mosaic effect
            if (toggleBtn) {
                toggleBtn.textContent = 'Disable Mosaic';
                toggleBtn.classList.add('bg-red-600');
                toggleBtn.classList.remove('bg-blue-600');
            }
            
            if (mosaicControls) {
                showElement(mosaicControls);
            }
            
            this.startProcessing();
            updateStatus('main-status', 'Mosaic effect enabled');
        } else {
            // Disable mosaic effect
            if (toggleBtn) {
                toggleBtn.textContent = 'Enable Mosaic';
                toggleBtn.classList.remove('bg-red-600');
                toggleBtn.classList.add('bg-blue-600');
            }
            
            if (mosaicControls) {
                hideElement(mosaicControls);
            }
            
            this.stopProcessing();
            updateStatus('main-status', 'Mosaic effect disabled');
        }
    }

    /**
     * Start processing video with mosaic effect
     */
    startProcessing() {
        if (!this.videoElement || !this.canvas || !this.ctx) return;
        
        this.isProcessing = true;
        
        // Show canvas and hide video
        this.canvas.style.display = 'block';
        this.videoElement.style.opacity = '0';
        
        // Set canvas size to match video
        this.canvas.width = this.videoElement.videoWidth || 640;
        this.canvas.height = this.videoElement.videoHeight || 480;
        
        // Start processing loop
        this.processFrame();
    }

    /**
     * Stop processing video
     */
    stopProcessing() {
        this.isProcessing = false;
        
        // Hide canvas and show video
        this.canvas.style.display = 'none';
        this.videoElement.style.opacity = '1';
        
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
        if (!this.isProcessing) return;
        
        // Draw the video frame to canvas
        this.ctx.drawImage(this.videoElement, 0, 0, this.canvas.width, this.canvas.height);
        
        // Apply mosaic effect
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
}
