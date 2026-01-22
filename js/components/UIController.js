/**
 * UI Controller Component
 * Manages UI state and interactions
 */

import { createComponent, showElement, hideElement, updateButton, updateStatus } from '../lib/componentUtils.js';

export class UIController {
    constructor() {
        this.currentView = 'camera';
        this.isMenuOpen = false;
    }

    /**
     * Initialize UI controller
     */
    initialize() {
        this.setupEventListeners();
        this.setupKeyboardShortcuts();
        this.loadRememberedFile();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Navigation buttons
        const cameraViewBtn = document.getElementById('camera-view-btn');
        if (cameraViewBtn) {
            cameraViewBtn.addEventListener('click', () => this.switchView('camera'));
        }

        const uploadViewBtn = document.getElementById('upload-view-btn');
        if (uploadViewBtn) {
            uploadViewBtn.addEventListener('click', () => this.switchView('upload'));
        }

        // Menu toggle button
        const menuToggleBtn = document.getElementById('menu-toggle-btn');
        if (menuToggleBtn) {
            menuToggleBtn.addEventListener('click', () => this.toggleMenu());
        }

        // View navigation
        const viewNavItems = document.querySelectorAll('.view-nav-item');
        viewNavItems.forEach(item => {
            item.addEventListener('click', (e) => {
                const view = e.currentTarget.dataset.view;
                if (view) {
                    this.switchView(view);
                }
            });
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            const menu = document.getElementById('side-menu');
            const menuToggle = document.getElementById('menu-toggle-btn');
            
            if (this.isMenuOpen && 
                menu && 
                !menu.contains(e.target) && 
                !menuToggle.contains(e.target)) {
                this.toggleMenu();
            }
        });

        // Window resize
        window.addEventListener('resize', () => this.handleResize());

        // Accordion functionality
        this.setupAccordion();

        // Handle toggleControlPanel event
        document.addEventListener('toggleControlPanel', () => this.toggleControlPanel());
    }

    /**
     * Setup accordion functionality for control panel sections
     */
    setupAccordion() {
        // Add event listeners to all accordion headers
        const accordionHeaders = document.querySelectorAll('.accordion-header');
        
        accordionHeaders.forEach(header => {
            header.addEventListener('click', () => {
                const section = header.parentElement;
                const content = header.nextElementSibling;
                const isOpen = !content.classList.contains('hidden');
                const icon = header.querySelector('.accordion-icon');

                // Close all accordion sections
                accordionHeaders.forEach(otherHeader => {
                    const otherContent = otherHeader.nextElementSibling;
                    const otherIcon = otherHeader.querySelector('.accordion-icon');
                    
                    otherContent.classList.add('hidden');
                    otherIcon.textContent = '+';
                });

                // Toggle current section if it was closed
                if (!isOpen) {
                    content.classList.remove('hidden');
                    icon.textContent = '-';
                }
            });
        });
    }

    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ignore if user is typing in an input field
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
                return;
            }

            // Toggle menu with Escape key
            if (e.key === 'Escape') {
                if (this.isMenuOpen) {
                    this.toggleMenu();
                }
            }

            // Switch views with number keys
            if (e.key === '1') {
                this.switchView('camera');
            } else if (e.key === '2') {
                this.switchView('upload');
            }

            // Camera shortcuts
            if (this.currentView === 'camera') {
                if (e.key === ' ') {
                    e.preventDefault();
                    // Toggle recording
                    const recordBtn = document.getElementById('record-video-btn');
                    if (recordBtn && !recordBtn.disabled) {
                        recordBtn.click();
                    }
                } else if (e.key === 'c') {
                    // Toggle camera
                    const cameraBtn = document.getElementById('toggle-camera-btn');
                    if (cameraBtn && !cameraBtn.disabled) {
                        cameraBtn.click();
                    }
                } else if (e.key === 's') {
                    // Switch camera
                    const switchBtn = document.getElementById('switch-camera-btn');
                    if (switchBtn && !switchBtn.disabled) {
                        switchBtn.click();
                    }
                } else if (e.key === 'p') {
                    // Toggle PiP
                    const pipBtn = document.getElementById('toggle-pip-btn');
                    if (pipBtn && !pipBtn.disabled) {
                        pipBtn.click();
                    }
                }
            }

            // Upload view shortcuts
            if (this.currentView === 'upload') {
                if (e.key === 'v') {
                    // Open video file dialog
                    const videoInput = document.getElementById('video-file-input');
                    if (videoInput) {
                        videoInput.click();
                    }
                } else if (e.key === 'i') {
                    // Open image file dialog
                    const imageInput = document.getElementById('image-file-input');
                    if (imageInput) {
                        imageInput.click();
                    }
                }
            }
        });
    }

    /**
     * Switch between views
     * @param {string} view - View to switch to ('camera' or 'upload')
     */
    switchView(view) {
        if (this.currentView === view) return;

        // Hide current view
        const currentViewElement = document.getElementById(`${this.currentView}-view`);
        if (currentViewElement) {
            hideElement(currentViewElement);
        }

        // Update navigation state
        const currentNavBtn = document.getElementById(`${this.currentView}-view-btn`);
        if (currentNavBtn) {
            currentNavBtn.classList.remove('bg-gray-700');
            currentNavBtn.classList.add('text-gray-400');
        }

        // Show new view
        const newViewElement = document.getElementById(`${view}-view`);
        if (newViewElement) {
            showElement(newViewElement);
        }

        // Update navigation state
        const newNavBtn = document.getElementById(`${view}-view-btn`);
        if (newNavBtn) {
            newNavBtn.classList.add('bg-gray-700');
            newNavBtn.classList.remove('text-gray-400');
        }

        // Update current view
        this.currentView = view;

        // Update status
        updateStatus('main-status', `Switched to ${view} view`);
    }

    /**
     * Toggle side menu
     */
    toggleMenu() {
        const menu = document.getElementById('side-menu');
        if (!menu) return;

        if (this.isMenuOpen) {
            hideElement(menu);
            this.isMenuOpen = false;
        } else {
            showElement(menu);
            this.isMenuOpen = true;
        }
    }

    /**
     * Handle window resize
     */
    handleResize() {
        // Close menu on small screens
        if (window.innerWidth < 768 && this.isMenuOpen) {
            this.toggleMenu();
        }
    }

    /**
     * Load remembered file from localStorage
     */
    loadRememberedFile() {
        const rememberedFile = localStorage.getItem('remembered-file');
        if (!rememberedFile) return;
        
        try {
            const file = JSON.parse(rememberedFile);
            const rememberedFileInfo = document.getElementById('remembered-file-info');
            
            if (rememberedFileInfo) {
                rememberedFileInfo.innerHTML = `
                    <div class="flex items-center justify-between">
                        <span class="text-sm">Last file: ${file.name}</span>
                        <button id="clear-remembered-file" class="text-xs text-gray-400 hover:text-white">
                            Clear
                        </button>
                    </div>
                `;
                
                // Add clear button event listener
                const clearBtn = document.getElementById('clear-remembered-file');
                if (clearBtn) {
                    clearBtn.addEventListener('click', () => {
                        localStorage.removeItem('remembered-file');
                        hideElement(rememberedFileInfo);
                    });
                }
                
                showElement(rememberedFileInfo);
            }
        } catch (error) {
            console.error('Error parsing remembered file:', error);
        }
    }

    /**
     * Show loading indicator
     * @param {string} message - Loading message
     */
    showLoading(message = 'Loading...') {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (!loadingOverlay) return;

        const loadingText = loadingOverlay.querySelector('.loading-text');
        if (loadingText) {
            loadingText.textContent = message;
        }

        showElement(loadingOverlay);
    }

    /**
     * Toggle control panel visibility
     */
    toggleControlPanel() {
        const controlPanelContainer = document.getElementById('control-panel-container');
        const toggleBtn = document.getElementById('toggle-control-panel-btn');
        const cameraViewContainer = document.getElementById('camera-view-container');
        const isHidden = controlPanelContainer.classList.contains('hidden');

        if (isHidden) {
            console.log("show control panel")
            // Show panel
            controlPanelContainer.classList.remove('hidden');
            if (toggleBtn) {
                toggleBtn.innerHTML = '← Hide Panel';
            }
        } else {
            console.log("hide control panel")
            // Hide panel
            controlPanelContainer.classList.add('hidden');
            if (toggleBtn) {
                toggleBtn.innerHTML = '→ Show Panel';
            }
        }
    }

    /**
     * Hide loading indicator
     */
    hideLoading() {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            hideElement(loadingOverlay);
        }
    }

    /**
     * Show error message
     * @param {string} message - Error message
     */
    showError(message) {
        const errorContainer = document.getElementById('error-container');
        if (!errorContainer) return;

        const errorMessage = errorContainer.querySelector('.error-message');
        if (errorMessage) {
            errorMessage.textContent = message;
        }

        showElement(errorContainer);

        // Auto-hide after 5 seconds
        setTimeout(() => {
            hideElement(errorContainer);
        }, 5000);
    }

    /**
     * Show success message
     * @param {string} message - Success message
     */
    showSuccess(message) {
        const successContainer = document.getElementById('success-container');
        if (!successContainer) return;

        const successMessage = successContainer.querySelector('.success-message');
        if (successMessage) {
            successMessage.textContent = message;
        }

        showElement(successContainer);

        // Auto-hide after 3 seconds
        setTimeout(() => {
            hideElement(successContainer);
        }, 3000);
    }

    /**
     * Update UI theme
     * @param {string} theme - Theme name ('light' or 'dark')
     */
    updateTheme(theme) {
        const body = document.body;
        
        if (theme === 'light') {
            body.classList.remove('dark');
            body.classList.add('light');
        } else {
            body.classList.remove('light');
            body.classList.add('dark');
        }
        
        // Save theme preference
        localStorage.setItem('theme', theme);
    }

    /**
     * Load saved theme
     */
    loadTheme() {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        this.updateTheme(savedTheme);
    }

    /**
     * Setup drag and drop for file uploads
     * @param {HTMLElement} dropZone - Drop zone element
     * @param {Function} onDrop - Callback function when files are dropped
     */
    setupDragAndDrop(dropZone, onDrop) {
        if (!dropZone) return;

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('bg-gray-700');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('bg-gray-700');
            });
        });

        // Handle dropped files
        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0 && onDrop) {
                onDrop(files);
            }
        });
    }
}
