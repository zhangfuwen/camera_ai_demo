/**
 * Food Detection Controller Component
 * Handles food detection from video and image files
 */

import { createComponent, showElement, hideElement, updateButton, updateStatus, createProgressBar, showNotification } from '../lib/componentUtils.js';

export class FoodDetectionController {
    constructor() {
        this.detectionResults = [];
        this.currentVideoFile = null;
        this.detectionInProgress = false;
        this.progressBar = null;
    }

    /**
     * Initialize food detection controller
     */
    initialize() {
        this.setupEventListeners();
        this.progressBar = createProgressBar('progress-container', 'detection-progress', 'progress-text');
        this.progressBar.hide();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        document.addEventListener("toggleFoodDetection", () => {
            console.log("Food detection toggled");
        });

        // Clear results button
        const clearResultsBtn = document.getElementById('clear-results-btn');
        if (clearResultsBtn) {
            clearResultsBtn.addEventListener('click', () => this.clearResults());
        }
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
            
            if (nameElement) nameElement.textContent = detection.name || 'Unknown';
            if (confidenceElement) confidenceElement.textContent = `${Math.round((detection.confidence || 0) * 100)}%`;
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
        
        if (modalTitle) modalTitle.textContent = detection.name || 'Unknown Food';
        
        if (modalBody) {
            modalBody.innerHTML = `
                <div class="space-y-4">
                    <div>
                        <h4 class="font-semibold text-sm text-gray-400">Confidence</h4>
                        <p>${Math.round((detection.confidence || 0) * 100)}%</p>
                    </div>
                    <div>
                        <h4 class="font-semibold text-sm text-gray-400">Estimated Calories</h4>
                        <p>${detection.calories || 'N/A'}</p>
                    </div>
                    <div>
                        <h4 class="font-semibold text-sm text-gray-400">Bounding Box</h4>
                        <p>X: ${detection.bbox?.x || 'N/A'}, Y: ${detection.bbox?.y || 'N/A'}</p>
                        <p>Width: ${detection.bbox?.width || 'N/A'}, Height: ${detection.bbox?.height || 'N/A'}</p>
                    </div>
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
