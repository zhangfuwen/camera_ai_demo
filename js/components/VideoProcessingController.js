/**
 * Video Processing Controller Component
 * Handles video file processing and management
 */

import { createComponent, showElement, hideElement, updateButton, updateStatus, showNotification, formatFileSize, createProgressBar } from '../lib/componentUtils.js';

export class VideoProcessingController {
    constructor() {
        this.currentVideoFile = null;
        this.processingInProgress = false;
        this.progressBar = null;
    }

    /**
     * Initialize video processing controller
     */
    initialize() {
        this.setupEventListeners();
        this.progressBar = createProgressBar('video-progress-container', 'video-processing-progress', 'video-progress-text');
        this.progressBar.hide();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Video file input
        const videoFileInput = document.getElementById('video-file-input');
        if (videoFileInput) {
            videoFileInput.addEventListener('change', (e) => this.handleVideoFileSelect(e));
        }

        // Clear video button
        const clearVideoBtn = document.getElementById('clear-video-btn');
        if (clearVideoBtn) {
            clearVideoBtn.addEventListener('click', () => this.clearVideo());
        }
    }

    /**
     * Handle video file selection
     * @param {Event} event - File input change event
     */
    handleVideoFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('video/')) {
            showNotification('Please select a valid video file', 'error');
            return;
        }

        this.currentVideoFile = file;
        this.displayVideoFile(file);

        // Remember file for next session
        localStorage.setItem('remembered-video-file', JSON.stringify({
            name: file.name,
            size: file.size,
            type: file.type,
            lastModified: file.lastModified
        }));

        // Show remembered file info
        this.displayRememberedFile(file);

        // Show clear button
        const clearBtn = document.getElementById('clear-video-btn');
        if (clearBtn) {
            showElement(clearBtn);
        }
        
        // Enable parse video button
        const parseBtn = document.getElementById('parse-video-btn');
        if (parseBtn) {
            parseBtn.disabled = false;
        }
    }

    /**
     * Display video file
     * @param {File} file - Video file
     */
    displayVideoFile(file) {
        const videoElement = document.getElementById('main-video');
        if (!videoElement) return;

        const url = URL.createObjectURL(file);
        videoElement.src = url;
        videoElement.type = file.type;
        
        showElement(videoElement);
        hideElement(document.getElementById('main-image'));
        
        updateStatus('main-status', `Loaded video: ${file.name} (${formatFileSize(file.size)})`);

        // Update file info display
        const fileInfo = document.getElementById('video-file-info');
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
     * Display processing results
     * @param {Object} results - Processing results from API
     */
    displayProcessingResults(results) {
        const resultsContainer = document.getElementById('video-processing-results');
        if (!resultsContainer) return;

        // Clear previous results
        resultsContainer.innerHTML = '';

        if (!results || !results.success) {
            resultsContainer.innerHTML = '<p class="text-gray-400 text-center py-4">Processing failed</p>';
            return;
        }

        // Create results header
        const header = document.createElement('div');
        header.className = 'flex justify-between items-center mb-4';
        header.innerHTML = `
            <h3 class="text-lg font-semibold">Processing Results</h3>
            <span class="text-sm text-gray-400">Completed</span>
        `;
        resultsContainer.appendChild(header);

        // Create results content
        const content = document.createElement('div');
        content.className = 'space-y-4';

        // Add video info
        if (results.videoInfo) {
            const videoInfoItem = createComponent('info-item');
            const titleElement = videoInfoItem.querySelector('.info-title');
            const valueElement = videoInfoItem.querySelector('.info-value');
            
            if (titleElement) titleElement.textContent = 'Video Information';
            if (valueElement) {
                valueElement.innerHTML = `
                    <div>Duration: ${results.videoInfo.duration || 'N/A'}</div>
                    <div>Resolution: ${results.videoInfo.resolution || 'N/A'}</div>
                    <div>Frame Rate: ${results.videoInfo.frameRate || 'N/A'}</div>
                `;
            }
            
            content.appendChild(videoInfoItem);
        }

        // Add processing info
        if (results.processingInfo) {
            const processingInfoItem = createComponent('info-item');
            const titleElement = processingInfoItem.querySelector('.info-title');
            const valueElement = processingInfoItem.querySelector('.info-value');
            
            if (titleElement) titleElement.textContent = 'Processing Information';
            if (valueElement) {
                valueElement.innerHTML = `
                    <div>Frames Processed: ${results.processingInfo.framesProcessed || 'N/A'}</div>
                    <div>Processing Time: ${results.processingInfo.processingTime || 'N/A'}</div>
                    <div>Output File: ${results.processingInfo.outputFile || 'N/A'}</div>
                `;
            }
            
            content.appendChild(processingInfoItem);
        }

        // Add download link if available
        if (results.downloadUrl) {
            const downloadItem = createComponent('download-item');
            const downloadBtn = downloadItem.querySelector('.download-btn');
            
            if (downloadBtn) {
                downloadBtn.href = results.downloadUrl;
                downloadBtn.download = results.downloadName || 'processed_video.mp4';
            }
            
            content.appendChild(downloadItem);
        }

        resultsContainer.appendChild(content);
        showElement(resultsContainer);
    }

    /**
     * Clear current video
     */
    clearVideo() {
        this.currentVideoFile = null;
        
        // Clear video element
        const videoElement = document.getElementById('main-video');
        if (videoElement) {
            videoElement.src = '';
            hideElement(videoElement);
        }
        
        // Clear file info
        const fileInfo = document.getElementById('video-file-info');
        if (fileInfo) {
            fileInfo.innerHTML = '';
            hideElement(fileInfo);
        }
        
        // Clear results
        const resultsContainer = document.getElementById('video-processing-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = '';
            hideElement(resultsContainer);
        }
        
        // Reset file input
        const fileInput = document.getElementById('video-file-input');
        if (fileInput) {
            fileInput.value = '';
        }
        
        // Hide clear button
        const clearBtn = document.getElementById('clear-video-btn');
        if (clearBtn) {
            hideElement(clearBtn);
        }
        
        // Disable parse video button
        const parseBtn = document.getElementById('parse-video-btn');
        if (parseBtn) {
            parseBtn.disabled = true;
        }
        
        updateStatus('main-status', 'Video cleared');
    }

    /**
     * Display remembered file info
     * @param {File} file - File to remember (optional)
     */
    displayRememberedFile(file = null) {
        const rememberedFileInfo = document.getElementById('remembered-file-info');
        if (!rememberedFileInfo) return;
        
        // If no file provided, try to get from localStorage
        if (!file) {
            const rememberedFile = localStorage.getItem('remembered-video-file');
            if (!rememberedFile) return;
            
            try {
                file = JSON.parse(rememberedFile);
            } catch (error) {
                console.error('Error parsing remembered file:', error);
                return;
            }
        }
        
        rememberedFileInfo.innerHTML = `
            <div class="flex items-center justify-between">
                <span class="text-sm">Last video: ${file.name}</span>
                <button id="clear-remembered-file" class="text-xs text-gray-400 hover:text-white">
                    Clear
                </button>
            </div>
        `;

        // Add clear button event listener
        const clearBtn = document.getElementById('clear-remembered-file');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                localStorage.removeItem('remembered-video-file');
                hideElement(rememberedFileInfo);
                
                // Disable parse video button when clearing remembered file
                const parseBtn = document.getElementById('parse-video-btn');
                if (parseBtn) {
                    parseBtn.disabled = true;
                }
            });
        }

        showElement(rememberedFileInfo);
        
        // Enable parse video button when there's a remembered file
        const parseBtn = document.getElementById('parse-video-btn');
        if (parseBtn) {
            parseBtn.disabled = false;
        }
    }
}
