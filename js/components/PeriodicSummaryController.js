/**
 * Periodic Summary Controller Component
 * Handles toggling periodic summarization functionality
 */

import { createComponent, showElement, hideElement, updateButton, updateStatus, showNotification } from '../lib/componentUtils.js';

export class PeriodicSummaryController {
    constructor() {
        this.isPeriodicSummaryActive = false;
        this.summaryButton = null;
        this.summaryStatus = null;
        this.socket = null;
    }

    /**
     * Initialize periodic summary controller
     */
    initialize() {
        this.setupEventListeners();
        this.summaryButton = document.getElementById('periodic-summary-btn');
        this.summaryStatus = document.getElementById('periodic-summary-status');
        
        // Initialize socket connection if not already done
        this.initializeSocket();
        
        // Show the status element
        if (this.summaryStatus) {
            // showElement(this.summaryStatus);
            updateStatus('periodic-summary-status', 'Periodic Summary: Off', 'info');
        }
    }

    /**
     * Initialize socket connection
     */
    initializeSocket() {
        if (typeof io !== 'undefined') {
            this.socket = io();
            
            // Listen for periodic summary status updates
            this.socket.on('periodic_summary_status', (data) => {
                this.handleStatusUpdate(data);
            });
            
            // Listen for periodic summary results
            this.socket.on('overall_status', (data) => {
                if (this.isPeriodicSummaryActive) {
                    console.log('Received periodic summary:', data);
                    // You can update UI here to show the summary
                }
            });
        } else {
            console.warn('Socket.IO not available, periodic summary will use HTTP fallback');
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        document.addEventListener("togglePeriodicSummary", () => {
            this.togglePeriodicSummary();
        });
    }

    /**
     * Toggle periodic summary on/off
     */
    async togglePeriodicSummary() {
        try {
            if (this.isPeriodicSummaryActive) {
                await this.stopPeriodicSummary();
            } else {
                await this.startPeriodicSummary();
            }
        } catch (error) {
            console.error('Error toggling periodic summary:', error);
            showNotification(`Error: ${error.message}`, 'error');
        }
    }

    /**
     * Start periodic summary
     */
    async startPeriodicSummary() {
        try {
            // Call backend API to start periodic summary
            const response = await fetch('/api/periodic-summary/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.isPeriodicSummaryActive = true;
                updateButton(this.summaryButton, 'Stop Periodic Summary', false);
                updateStatus('periodic-summary-status', 'Periodic Summary: Active', 'success');
                showNotification('Periodic summary started', 'success');
            } else {
                throw new Error(result.error || 'Failed to start periodic summary');
            }
        } catch (error) {
            console.error('Error starting periodic summary:', error);
            throw error;
        }
    }

    /**
     * Stop periodic summary
     */
    async stopPeriodicSummary() {
        try {
            // Call backend API to stop periodic summary
            const response = await fetch('/api/periodic-summary/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.isPeriodicSummaryActive = false;
                updateButton(this.summaryButton, 'Start Periodic Summary', false);
                updateStatus('periodic-summary-status', 'Periodic Summary: Off', 'info');
                showNotification('Periodic summary stopped', 'info');
            } else {
                throw new Error(result.error || 'Failed to stop periodic summary');
            }
        } catch (error) {
            console.error('Error stopping periodic summary:', error);
            throw error;
        }
    }

    /**
     * Handle status updates from backend
     */
    handleStatusUpdate(data) {
        if (data.active !== this.isPeriodicSummaryActive) {
            this.isPeriodicSummaryActive = data.active;
            
            if (data.active) {
                updateButton(this.summaryButton, 'Stop Periodic Summary', false);
                updateStatus('periodic-summary-status', 'Periodic Summary: Active', 'success');
            } else {
                updateButton(this.summaryButton, 'Start Periodic Summary', false);
                updateStatus('periodic-summary-status', 'Periodic Summary: Off', 'info');
            }
        }
    }

    /**
     * Get current status
     */
    getStatus() {
        return {
            active: this.isPeriodicSummaryActive,
            connected: this.socket && this.socket.connected
        };
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.socket) {
            this.socket.disconnect();
        }
    }
}
