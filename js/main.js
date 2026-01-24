/**
 * Main application entry point
 * Handles UI initialization and component coordination
 */

// Import component modules
import { createComponent, updateButton, updateStatus, showNotification } from './lib/componentUtils.js';
import { CameraController } from './components/CameraController.js';
import { FoodDetectionController } from './components/FoodDetectionController.js';
import { CalorieEstimationController } from './components/CalorieEstimationController.js';
import { VideoProcessingController } from './components/VideoProcessingController.js';
import { ImageProcessingController } from './components/ImageProcessingController.js';
import { UIController } from './components/UIController.js';
import { LLMClient } from './components/LLMClient.js';
import { MosaicEffectController } from './components/MosaicEffectController.js';

// Application state - exposed to window for component communication
window.app = {
    cameraController: null,
    foodDetectionController: null,
    calorieEstimationController: null,
    videoProcessingController: null,
    imageProcessingController: null,
    uiController: null,
    llmClient: null,
    mosaicEffectController: null,
    isInitialized: false
};

// For compatibility, also keep the app variable
const app = window.app;

/**
 * Application initialization
 */
async function init() {
    try {
        // Create UI components from templates
        createUIComponents();
        
        // Initialize UI controller first
        app.uiController = new UIController();
        app.uiController.initialize();
        
        // Initialize controllers
        app.cameraController = new CameraController();
        app.foodDetectionController = new FoodDetectionController();
        app.calorieEstimationController = new CalorieEstimationController();
        app.videoProcessingController = new VideoProcessingController();
        app.imageProcessingController = new ImageProcessingController();
        app.llmClient = new LLMClient();
        app.mosaicEffectController = new MosaicEffectController();
        
        // Initialize components
        await initializeComponents();
        
        // Set up event delegation
    setupEventDelegation();
    
    // Set up custom event listeners
    setupCustomEventListeners();
    
    // Initialize camera permission
    await app.cameraController.initializeCameraPermission();
        
        // Load remembered video file information
        app.videoProcessingController.displayRememberedFile();

        // Mark as initialized
        app.isInitialized = true;
        
        console.log('Application initialized successfully');
    } catch (error) {
        console.error('Failed to initialize application:', error);
        showError('Application initialization failed: ' + error.message);
    }
}

/**
 * Initialize UI components from templates
 */
async function initializeComponents() {
    // Get video element for camera controller
    const videoElement = document.getElementById('main-video');
    if (videoElement) {
        await app.cameraController.initialize(videoElement);
    }

    // Initialize food detection controller
    app.foodDetectionController.initialize();

    // Initialize calorie estimation controller
    app.calorieEstimationController.initialize();

    // Initialize video processing controller
    app.videoProcessingController.initialize();
    
    // Initialize image processing controller
    app.imageProcessingController.initialize();
    
    // Initialize LLM client
    app.llmClient.initialize();
    
    // Initialize mosaic effect controller
    app.mosaicEffectController.initialize();
}

/**
 * Create UI components from templates
 */
function createUIComponents() {
    // Create control panel
    const controlPanelTemplate = document.getElementById('control-panel-template');
    if (!controlPanelTemplate) {
        throw new Error('Control panel template not found');
    }
    
    const controlPanel = createComponent('control-panel-template');
    const controlPanelContainer = document.getElementById('control-panel-container');
    controlPanelContainer.appendChild(controlPanel);
    
    // Create camera view
    const cameraViewTemplate = document.getElementById('camera-view-template');
    if (!cameraViewTemplate) {
        throw new Error('Camera view template not found');
    }
    
    const cameraView = createComponent('camera-view-template');
    const cameraViewContainer = document.getElementById('camera-view-container');
    cameraViewContainer.appendChild(cameraView);
}

/**
 * Set up event delegation for dynamic content
 */
function setupEventDelegation() {
    document.addEventListener('click', (event) => {
        console.log("Event target:", event.target);
        const target = event.target;
        
        // Request camera permission
        if (target.matches('#request-permission-btn')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('requestPermission'));
        }
        
        // Main camera control - toggle button
        if (target.matches('#toggle-main-btn')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('toggleCamera', { detail: { type: 'main' } }));
        }
        
        // PIP camera control - toggle button
        if (target.matches('#toggle-pip-camera-btn')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('toggleCamera', { detail: { type: 'pip' } }));
        }
        
        // Food detection controls
        if (target.matches('#food-detection-btn')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('toggleFoodDetection'));
        }
        
        // Mosaic effect controls
        if (target.matches('#toggle-mosaic-btn')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('toggleMosaicEffect'));
        }
        
        // Calorie estimation controls
        if (target.matches('#calorie-estimation-btn')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('toggleCalorieEstimation'));
        }
        
        // PIP controls
        if (target.matches('#toggle-pip-btn')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('togglePipVisibility'));
        }
        
        if (target.matches('#swap-cameras-btn')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('swapCameras'));
        }
        
        // PIP positioning controls
        if (target.matches('#pip-up')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('movePip', { detail: { direction: 'up' } }));
        }
        
        if (target.matches('#pip-down')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('movePip', { detail: { direction: 'down' } }));
        }
        
        if (target.matches('#pip-left')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('movePip', { detail: { direction: 'left' } }));
        }
        
        if (target.matches('#pip-right')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('movePip', { detail: { direction: 'right' } }));
        }
        
        // Control panel toggle
        if (target.matches('#toggle-control-panel-btn')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('toggleControlPanel'));
        }
        
        // Video processing controls
        if (target.matches('#parse-video-btn')) {
            console.log("Video analysis button clicked");
            event.preventDefault();
            // Get the current video file from VideoProcessingController
            const videoFile = window.app.videoProcessingController.currentVideoFile;
            if (!videoFile) {
                showNotification('Please select a video file first', 'error');
                return;
            }
            
            // // Update UI
            // const parseBtn = document.getElementById('video-analysis-btn');
            // if (parseBtn) {
            //     updateButton(parseBtn, 'Testing LLM...', true);
            // }
            
            // Call LLM client's testVideoWithLLM method
            window.app.llmClient.testVideoWithLLM(videoFile)
                .then(result => {
                    // Display results using LLM client
                    window.app.llmClient.displayAnalysisResults(result);
                    
                    // Update status
                    updateStatus('main-status', 'LLM video analysis completed successfully');
                    showNotification('LLM video analysis completed', 'success');
                })
                .catch(error => {
                    console.error('Error processing video with LLM:', error);
                    updateStatus('main-status', `Error: ${error.message}`);
                    showNotification(`Error: ${error.message}`, 'error');
                })
                .finally(() => {
                    // Reset button
                    if (parseBtn) {
                        updateButton(parseBtn, 'LLM Video Test', false);
                    }
                });
        }
        
        if (target.matches('#record-video-btn')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('toggleVideoRecording'));
        }
        
        if (target.matches('#save-video-btn')) {
            event.preventDefault();
            document.dispatchEvent(new CustomEvent('saveRecordedVideo'));
        }
    });
    
    // File input change event
    document.addEventListener('change', (event) => {
        if (event.target.matches('#video-file-input')) {
            document.dispatchEvent(new CustomEvent('videoFileSelected', { 
                detail: { file: event.target.files[0] } 
            }));
        }
        
        if (event.target.matches('#image-file-input')) {
            document.dispatchEvent(new CustomEvent('imageFileSelected', { 
                detail: { file: event.target.files[0] } 
            }));
        }
        
        // Camera selection change events
        if (event.target.matches('#main-camera-select')) {
            document.dispatchEvent(new CustomEvent('mainCameraChanged', { 
                detail: { deviceId: event.target.value } 
            }));
        }
        
        if (event.target.matches('#pip-camera-select')) {
            document.dispatchEvent(new CustomEvent('pipCameraChanged', { 
                detail: { deviceId: event.target.value } 
            }));
        }
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (event) => {
        // Check if focus is on an input field
        const activeElement = document.activeElement;
        const isInputField = activeElement.tagName === 'INPUT' || 
                            activeElement.tagName === 'TEXTAREA' || 
                            activeElement.contentEditable === 'true';
        
        // If focus is on an input field, don't handle keyboard shortcuts
        if (isInputField) return;
        
        // Handle keyboard shortcuts
        switch (event.key) {
            case '1':
                event.preventDefault();
                document.dispatchEvent(new CustomEvent('toggleMainCamera'));
                break;
            case '2':
                event.preventDefault();
                document.dispatchEvent(new CustomEvent('togglePipCamera'));
                break;
            case 'p':
            case 'P':
                event.preventDefault();
                document.dispatchEvent(new CustomEvent('togglePipVisibility'));
                break;
            case 's':
            case 'S':
                event.preventDefault();
                document.dispatchEvent(new CustomEvent('swapCameras'));
                break;
            case 'f':
            case 'F':
                event.preventDefault();
                document.dispatchEvent(new CustomEvent('toggleFoodDetection'));
                break;
            case 'm':
            case 'M':
                event.preventDefault();
                document.dispatchEvent(new CustomEvent('toggleMosaicEffect'));
                break;
            case 'c':
            case 'C':
                event.preventDefault();
                document.dispatchEvent(new CustomEvent('toggleCalorieEstimation'));
                break;
            case 'ArrowUp':
                event.preventDefault();
                document.dispatchEvent(new CustomEvent('movePip', { detail: { direction: 'up' } }));
                break;
            case 'ArrowDown':
                event.preventDefault();
                document.dispatchEvent(new CustomEvent('movePip', { detail: { direction: 'down' } }));
                break;
            case 'ArrowLeft':
                event.preventDefault();
                document.dispatchEvent(new CustomEvent('movePip', { detail: { direction: 'left' } }));
                break;
            case 'ArrowRight':
                event.preventDefault();
                document.dispatchEvent(new CustomEvent('movePip', { detail: { direction: 'right' } }));
                break;
        }
    });
    
    // Drag and drop events
    setupDragAndDrop();
}

/**
 * Set up custom event listeners for component communication
 */
function setupCustomEventListeners() {
    // Mosaic effect toggle
    document.addEventListener('toggleMosaicEffect', () => {
        if (app.mosaicEffectController) {
            app.mosaicEffectController.toggleMosaicEffect();
        }
    });
    
    document.addEventListener('togglePipView', () => {
        if (app.cameraController) {
            app.cameraController.togglePipView();
        }
    });
    
    // Rotation and mirror controls
    document.addEventListener('setRotation', (e) => {
        if (app.cameraController) {
            app.cameraController.setRotation(e.detail.angle);
        }
    });
    
    document.addEventListener('toggleMirror', () => {
        if (app.cameraController) {
            app.cameraController.toggleMirror();
        }
    });
}

/**
 * Setup drag and drop events
 */
function setupDragAndDrop() {
    const mainVideo = document.getElementById('main-video');
    const pipVideo = document.getElementById('pip-video');
    
    [mainVideo, pipVideo].forEach(videoElement => {
        videoElement.addEventListener('dragover', (event) => {
            event.preventDefault();
            videoElement.classList.add('border-4', 'border-blue-500');
        });
        
        videoElement.addEventListener('dragleave', (event) => {
            event.preventDefault();
            videoElement.classList.remove('border-4', 'border-blue-500');
        });
        
        videoElement.addEventListener('drop', (event) => {
            event.preventDefault();
            videoElement.classList.remove('border-4', 'border-blue-500');
            
            const files = event.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('video/')) {
                    document.dispatchEvent(new CustomEvent('videoFileDropped', { 
                        detail: { file, target: videoElement.id } 
                    }));
                } else {
                    showError('Please drop a video file');
                }
            }
        });
    });
}

/**
 * Display error message to user
 */
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'fixed top-4 right-4 bg-red-600 text-white p-4 rounded-lg shadow-lg z-50';
    errorDiv.textContent = message;
    
    document.body.appendChild(errorDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentNode) {
            errorDiv.parentNode.removeChild(errorDiv);
        }
    }, 5000);
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', init);
