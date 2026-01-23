/**
 * Image Processing Controller Component
 * Handles image file processing and management
 */

import { createComponent, showElement, hideElement, updateButton, updateStatus, showNotification, formatFileSize } from '../lib/componentUtils.js';

export class ImageProcessingController {
    constructor() {
        this.currentImageFile = null;
        this.processingInProgress = false;
    }

    /**
     * Initialize image processing controller
     */
    initialize() {
        this.setupEventListeners();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Image file input
        const imageFileInput = document.getElementById('image-file-input');
        if (imageFileInput) {
            imageFileInput.addEventListener('change', (e) => this.handleImageFileSelect(e));
        }

        // Clear image button
        const clearImageBtn = document.getElementById('clear-image-btn');
        if (clearImageBtn) {
            clearImageBtn.addEventListener('click', () => this.clearImage());
        }

        // Custom event for image file selection
        document.addEventListener('imageFileSelected', (e) => {
            this.handleImageFileSelect({ target: { files: [e.detail.file] } });
        });
    }

    /**
     * Handle image file selection
     * @param {Event} event - File input change event
     */
    handleImageFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            showNotification('Please select a valid image file', 'error');
            return;
        }

        this.currentImageFile = file;
        this.displayImageFile(file);

        // Show clear button
        const clearBtn = document.getElementById('clear-image-btn');
        if (clearBtn) {
            showElement(clearBtn);
        }
    }

    /**
     * Display image file
     * @param {File} file - Image file
     */
    displayImageFile(file) {
        const imageElement = document.getElementById('main-image');
        const videoElement = document.getElementById('main-video');
        const detectionOverlay = document.getElementById('detection-overlay');
        
        if (!imageElement) return;

        // Create object URL for the image
        const url = URL.createObjectURL(file);
        
        // Load the image to get its dimensions
        const img = new Image();
        img.onload = () => {
            // Set the image dimensions to match the actual image resolution
            imageElement.width = img.width;
            imageElement.height = img.height;
            
            // Set the canvas dimensions to match the image resolution
            if (detectionOverlay) {
                detectionOverlay.width = img.width;
                detectionOverlay.height = img.height;
                
                // Set canvas display size to match the image element's display size
                const imageRect = imageElement.getBoundingClientRect();
                detectionOverlay.style.width = `${imageRect.width}px`;
                detectionOverlay.style.height = `${imageRect.height}px`;
            }
            
            // Hide video and show image
            if (videoElement) {
                hideElement(videoElement);
            }
            showElement(imageElement);
            
            // Show detection overlay
            if (detectionOverlay) {
                showElement(detectionOverlay);
            }
            
            updateStatus('main-status', `Loaded image: ${file.name} (${formatFileSize(file.size)}) - Resolution: ${img.width}x${img.height}`);
            
            // Clean up the object URL
            URL.revokeObjectURL(url);
        };
        
        img.src = url;
        
        // Update file info display
        const fileInfo = document.getElementById('image-file-info');
        if (fileInfo) {
            fileInfo.innerHTML = `
                <div class="flex justify-between items-center">
                    <span class="text-sm">${file.name}</span>
                    <span class="text-xs text-gray-400">${formatFileSize(file.size)}</span>
                </div>
            `;
            showElement(fileInfo);
        }
    }

    /**
     * Clear current image
     */
    clearImage() {
        this.currentImageFile = null;
        
        // Clear image element
        const imageElement = document.getElementById('main-image');
        if (imageElement) {
            imageElement.src = '';
            hideElement(imageElement);
        }
        
        // Clear file input
        const imageInput = document.getElementById('image-file-input');
        if (imageInput) {
            imageInput.value = '';
        }
        
        // Hide file info
        const fileInfo = document.getElementById('image-file-info');
        if (fileInfo) {
            hideElement(fileInfo);
        }
        
        // Hide clear button
        const clearBtn = document.getElementById('clear-image-btn');
        if (clearBtn) {
            hideElement(clearBtn);
        }
        
        // Clear detection overlay
        const detectionOverlay = document.getElementById('detection-overlay');
        if (detectionOverlay) {
            const ctx = detectionOverlay.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, detectionOverlay.width, detectionOverlay.height);
            }
        }
        
        updateStatus('main-status', 'Image cleared');
    }
}
