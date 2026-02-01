/**
 * Component utilities for creating and managing UI components
 */

import { globalLogger } from '../utils/logger.js';

/**
 * Create a component from a template
 * @param {string} templateId - ID of the template element
 * @returns {DocumentFragment} - The cloned template content
 */
export function createComponent(templateId) {
    const tpl = document.getElementById(templateId);
    if (!tpl) {
        throw new Error(`Template #${templateId} not found`);
    }
    return document.importNode(tpl.content, true);
}

/**
 * Load an external HTML partial
 * @param {string} selector - CSS selector for element to replace
 * @param {string} path - Path to the HTML partial file
 * @returns {Promise<void>}
 */
export async function loadPartial(selector, path) {
    try {
        const res = await fetch(path);
        if (!res.ok) {
            throw new Error(`Failed to fetch partial: ${res.status} ${res.statusText}`);
        }
        
        const html = await res.text();
        const range = document.createRange();
        range.selectNode(document.body);
        const fragment = range.createContextualFragment(html);
        
        const element = document.querySelector(selector);
        if (!element) {
            throw new Error(`Element not found: ${selector}`);
        }
        
        element.replaceWith(fragment);
    } catch (error) {
        console.error('Error loading partial:', error);
        throw error;
    }
}

/**
 * Show an element by removing the 'hidden' class
 * @param {string|Element} element - Element or selector to show
 */
export function showElement(element) {
    globalLogger.info('showElement', element);
    const el = typeof element === 'string' ? document.querySelector(element) : element;
    if (el) {
        el.classList.remove('hidden');
    }
}

/**
 * Hide an element by adding the 'hidden' class
 * @param {string|Element} element - Element or selector to hide
 */
export function hideElement(element) {
    const el = typeof element === 'string' ? document.querySelector(element) : element;
    if (el) {
        el.classList.add('hidden');
    }
}

/**
 * Toggle element visibility
 * @param {string|Element} element - Element or selector to toggle
 */
export function toggleElement(element) {
    const el = typeof element === 'string' ? document.querySelector(element) : element;
    if (el) {
        el.classList.toggle('hidden');
    }
}

/**
 * Update button text and state
 * @param {string|Element} button - Button element or selector
 * @param {string} text - New button text
 * @param {boolean} disabled - Whether button should be disabled
 */
export function updateButton(button, text, disabled = false) {
    const btn = typeof button === 'string' ? document.querySelector(button) : button;
    if (btn) {
        btn.textContent = text;
        btn.disabled = disabled;
    }
}

/**
 * Update status text
 * @param {string|Element} statusElement - Status element or selector
 * @param {string} text - New status text
 */
export function updateStatus(statusElement, text) {
    const el = typeof statusElement === 'string' ? document.querySelector(statusElement) : statusElement;
    if (el) {
        el.textContent = text;
    }
}

/**
 * Show a notification message
 * @param {string} message - Message to display
 * @param {string} type - Type of notification (success, error, warning, info)
 * @param {number} duration - Duration in milliseconds to show notification
 */
export function showNotification(message, type = 'info', duration = 5000) {
    const notification = document.createElement('div');
    
    // Set classes based on type
    let bgClass = 'bg-blue-600'; // Default info
    if (type === 'success') bgClass = 'bg-green-600';
    else if (type === 'error') bgClass = 'bg-red-600';
    else if (type === 'warning') bgClass = 'bg-yellow-600';
    
    notification.className = `fixed top-4 right-4 ${bgClass} text-white p-4 rounded-lg shadow-lg z-50 max-w-md`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, duration);
}

/**
 * Format file size in human-readable format
 * @param {number} bytes - File size in bytes
 * @returns {string} - Formatted file size
 */
export function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Create a progress bar
 * @param {string} containerId - ID of container element
 * @param {string} progressBarId - ID for progress bar element
 * @param {string} progressTextId - ID for progress text element
 * @returns {Object} - Object with updateProgress method
 */
export function createProgressBar(containerId, progressBarId, progressTextId) {
    const container = document.getElementById(containerId);
    
    if (!container) {
        throw new Error(`Container element not found: ${containerId}`);
    }
    
    // Create progress bar HTML
    container.innerHTML = `
        <div class="w-full bg-gray-700 rounded">
            <div id="${progressBarId}" class="bg-green-500 h-5 rounded text-center leading-5 text-white" style="width: 0%">0%</div>
        </div>
        <div id="${progressTextId}" class="mt-2 text-xs"></div>
    `;
    
    // Return object with update method
    return {
        updateProgress(percent, text) {
            const progressBar = document.getElementById(progressBarId);
            const progressText = document.getElementById(progressTextId);
            
            if (progressBar) {
                progressBar.style.width = `${percent}%`;
                progressBar.textContent = `${percent}%`;
            }
            
            if (progressText) {
                progressText.textContent = text;
            }
        },
        
        show() {
            container.classList.remove('hidden');
        },
        
        hide() {
            container.classList.add('hidden');
        }
    };
}
