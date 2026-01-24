/**
 * Camera Controller Component
 * Handles camera operations, video recording, and camera switching
 */

import { createComponent, showElement, hideElement, updateButton, updateStatus } from '../lib/componentUtils.js';

export class CameraController {
    constructor() {
        this.videoElement = null;
        this.stream = null;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.isRecording = false;
        this.devices = [];
        this.currentDeviceId = null;
        this.mainCameraDeviceId = null;
        this.pipCameraDeviceId = null;
        this.pipVideo = null;
        this.pipVideoContainer = null;
        this.pipPosition = { x: 10, y: 10 };
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        // 添加旋转和镜像状态变量
        this.rotationAngle = 0;
        this.isMirrored = false;
        // Color adjustment properties
        this.brightness = 100;
        this.contrast = 100;
        this.saturation = 100;
        this.redChannel = 100;
        this.greenChannel = 100;
        this.blueChannel = 100;
        // Resolution settings
        this.desiredResolution = null;
    }

    /**
     * Initialize camera permission
     */
    async initializeCameraPermission() {
        try {
            // Request camera permission
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            // Stop the stream immediately after getting permission
            stream.getTracks().forEach(track => track.stop());
            
            updateStatus('main-status', 'Camera permission granted');
            this.hidePermissionPrompt();
            this.loadCameraDevices();
            return true;
        } catch (error) {
            console.error('Error requesting camera permission:', error);
            updateStatus('main-status', 'Error: Camera permission denied');
            return false;
        }
    }

    /**
     * Initialize camera controller
     * @param {HTMLElement} videoElement - Video element for camera display
     */
    async initialize(videoElement) {
        this.videoElement = videoElement;
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Set up UI controls
        this.setupTransformControls();
        this.setupColorAdjustmentControls();
        
        // Initialize rotation and mirror states
        this.rotationAngle = 0;
        this.isMirrored = false;
        
        // Load all saved camera settings from local storage
        this.loadAllSettings();
        
        // Initialize camera
        await this.startCamera();
    }

    /**
     * Get available camera devices and their capabilities
     */
    async getCameraDevices() {
        try {
            const allDevices = await navigator.mediaDevices.enumerateDevices();
            this.devices = allDevices.filter(device => device.kind === 'videoinput');
            
            if (this.devices.length > 0) {
                this.currentDeviceId = this.devices[0].deviceId;
                
                // Get capabilities for the first camera
                await this.getCameraCapabilities(this.devices[0].deviceId);
            }
            
            // Reset resolution selector when camera devices are loaded
            const resolutionSelect = document.getElementById('resolution-select');
            if (resolutionSelect) {
                resolutionSelect.value = '';
                resolutionSelect.disabled = this.devices.length === 0;
            }
            
            updateStatus('main-status', `Found ${this.devices.length} camera(s)`);
        } catch (error) {
            console.error('Error getting camera devices:', error);
            updateStatus('main-status', 'Error: Could not access camera devices');
        }
    }

    /**
     * Get camera capabilities including supported resolutions
     * @param {string} deviceId - Camera device ID
     */
    async getCameraCapabilities(deviceId) {
        try {
            // Get a temporary stream to check capabilities
            const tempStream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: { exact: deviceId } }
            });
            
            const videoTrack = tempStream.getVideoTracks()[0];
            const capabilities = videoTrack.getCapabilities();
            
            console.log('Camera capabilities:', capabilities);
            
            // Check if width and height ranges are available
            if (capabilities.width && capabilities.height) {
                console.log(`Supported resolution range: ${capabilities.width.min}x${capabilities.height.min} to ${capabilities.width.max}x${capabilities.height.max}`);
                console.log(`Resolution step: ${capabilities.width.step}x${capabilities.height.step}`);
                
                // Generate common resolutions within the supported range
                const commonResolutions = [
                    { width: 640, height: 480 },   // VGA
                    { width: 1280, height: 720 },  // 720p HD
                    { width: 1920, height: 1080 }, // 1080p Full HD
                    { width: 3840, height: 2160 }  // 4K UHD
                ];
                
                const supportedResolutions = commonResolutions.filter(res => 
                    res.width >= capabilities.width.min && 
                    res.width <= capabilities.width.max &&
                    res.height >= capabilities.height.min && 
                    res.height <= capabilities.height.max
                );

                console.log("capabilities:", capabilities);
                
                console.log('Supported common resolutions:', supportedResolutions);
                
                // Store supported resolutions for later use
                this.supportedResolutions = supportedResolutions;
                
                // Update UI with resolution options
                this.updateResolutionOptions(supportedResolutions);
            }
            
            // Stop the temporary stream
            tempStream.getTracks().forEach(track => track.stop());
        } catch (error) {
            console.error('Error getting camera capabilities:', error);
        }
    }

    /**
     * Update the UI with resolution options
     * @param {Array} resolutions - Array of supported resolutions
     */
    updateResolutionOptions(resolutions) {
        const resolutionSelect = document.getElementById('resolution-select');
        if (!resolutionSelect) return;
        
        // Clear existing options
        resolutionSelect.innerHTML = '<option value="">-- Default resolution --</option>';
        
        // Add supported resolutions
        resolutions.forEach(res => {
            const option = document.createElement('option');
            option.value = `${res.width}x${res.height}`;
            option.textContent = `${res.width}x${res.height} (${this.getResolutionName(res.width, res.height)})`;
            resolutionSelect.appendChild(option);
        });
        
        // Enable the selector
        resolutionSelect.disabled = false;
        
        // Log available options
        console.log('Available resolution options:');
        resolutions.forEach((res, index) => {
            console.log(`${index + 1}. ${res.width}x${res.height}`);
        });
    }

    /**
     * Get a user-friendly name for a resolution
     * @param {number} width - Resolution width
     * @param {number} height - Resolution height
     * @returns {string} - Resolution name
     */
    getResolutionName(width, height) {
        if (width === 640 && height === 480) return 'VGA';
        if (width === 1280 && height === 720) return '720p HD';
        if (width === 1920 && height === 1080) return '1080p Full HD';
        if (width === 3840 && height === 2160) return '4K UHD';
        return 'Custom';
    }

    /**
     * Set camera resolution
     * @param {number} width - Desired width
     * @param {number} height - Desired height
     */
    setCameraResolution(width, height) {
        this.desiredResolution = { width, height };
        console.log(`Camera resolution set to: ${width}x${height}`);
    }

    /**
     * Apply transforms to video element (rotation and mirror)
     */
    applyTransforms() {
        if (!this.videoElement) return;
        
        let transformValue = '';
        
        // Apply mirror effect
        if (this.isMirrored) {
            transformValue += 'scaleX(-1) ';
        }
        
        // Apply rotation
        if (this.rotationAngle !== 0) {
            transformValue += `rotate(${this.rotationAngle}deg) `;
        }
        
        // Apply transform to video element
        this.videoElement.style.transform = transformValue.trim() || 'none';
    }

    /**
     * Set rotation angle for video element
     * @param {number} angle - Rotation angle in degrees (0, 90, 180, 270)
     */
    setRotation(angle) {
        if ([0, 90, 180, 270].includes(angle)) {
            this.rotationAngle = angle;
            this.applyTransforms();
            this.saveAllSettings();
            updateStatus('main-status', `Main Camera: Active - Rotation: ${angle}°`);
        }
    }

    /**
     * Toggle mirror effect for video element
     */
    toggleMirror() {
        this.isMirrored = !this.isMirrored;
        this.applyTransforms();
        this.saveAllSettings();
        
        // Update UI button text
        const mirrorToggle = document.getElementById('mirror-toggle');
        const mirrorStatus = document.getElementById('mirror-status');
        if (mirrorStatus) {
            mirrorStatus.textContent = this.isMirrored ? 'On' : 'Off';
        }
        
        if (mirrorToggle) {
            mirrorToggle.classList.toggle('bg-gray-600', !this.isMirrored);
            mirrorToggle.classList.toggle('bg-blue-600', this.isMirrored);
        }
        
        updateStatus('main-status', `Main Camera: Active - Mirror: ${this.isMirrored ? 'On' : 'Off'}`);
    }

    /**
     * Setup transform controls event listeners
     */
    setupTransformControls() {
        // Rotation select
        const rotationSelect = document.getElementById('rotation-select');
        if (rotationSelect) {
            rotationSelect.addEventListener('change', (e) => {
                const angle = parseInt(e.target.value);
                this.setRotation(angle);
            });
        }

        // Mirror toggle button
        const mirrorToggle = document.getElementById('mirror-toggle');
        if (mirrorToggle) {
            mirrorToggle.addEventListener('click', () => {
                this.toggleMirror();
            });
        }
        
        // Enable the transform controls when camera permission is granted
        const transformControls = document.getElementById('transform-controls');
        if (transformControls) {
            const rotationSelect = document.getElementById('rotation-select');
            const mirrorToggle = document.getElementById('mirror-toggle');
            
            if (rotationSelect) rotationSelect.disabled = false;
            if (mirrorToggle) mirrorToggle.disabled = false;
        }
    }

    /**
     * Start camera with current device
     * @param {string} type - Camera type ('main' or 'pip')
     */
    async startCamera(type = 'main') {
        try {
            // Get selected device ID
            let deviceId = null;
            const selectElement = document.getElementById(type === 'main' ? 'main-camera-select' : 'pip-camera-select');
            
            // First, try to get the device ID from the dropdown selection
            if (selectElement && selectElement.value) {
                deviceId = selectElement.value;
            } else {
                // If no selection is made, try to use the saved camera selection
                if (type === 'main') {
                    deviceId = this.mainCameraDeviceId;
                } else {
                    deviceId = this.pipCameraDeviceId;
                }
                
                // If we have a saved device ID, set it in the dropdown
                if (deviceId && selectElement) {
                    selectElement.value = deviceId;
                }
            }
            
            if (!deviceId) {
                updateStatus(`${type}-status`, `Error: No camera selected`);
                return;
            }
            
            // Stop existing stream if it exists
            if (type === 'main' && this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            } else if (type === 'pip' && this.pipStream) {
                this.pipStream.getTracks().forEach(track => track.stop());
            }

            // Set up constraints with desired resolution if available
            const constraints = {
                video: { 
                    deviceId: { exact: deviceId },
                    ...(this.desiredResolution && {
                        width: { ideal: this.desiredResolution.width },
                        height: { ideal: this.desiredResolution.height }
                    })
                }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Get video track to check resolution
            const videoTrack = stream.getVideoTracks()[0];
            const settings = videoTrack.getSettings();
            console.log(`Camera resolution: ${settings.width}x${settings.height}`);
            console.log(stream.getVideoTracks().length)
            
            // Assign stream to the appropriate video element
            if (type === 'main') {
                this.stream = stream;
                this.videoElement.srcObject = stream;
                
                // Set video element dimensions to match the actual camera resolution
                this.videoElement.width = settings.width;
                this.videoElement.height = settings.height;
                
                // Also update the detection overlay canvas to match
                const detectionOverlay = document.getElementById('detection-overlay');
                if (detectionOverlay) {
                    detectionOverlay.width = settings.width;
                    detectionOverlay.height = settings.height;
                }
                
                // Update the main image element to match as well
                const mainImage = document.getElementById('main-image');
                if (mainImage) {
                    mainImage.width = settings.width;
                    mainImage.height = settings.height;
                }
                
                updateStatus('main-status', `Main Camera: Active (${settings.width}x${settings.height})`);
                
                // Enable food detection button when main camera is active
                const foodDetectionBtn = document.getElementById('food-detection-btn');
                if (foodDetectionBtn) {
                    foodDetectionBtn.disabled = false;
                }
                
                // Enable mosaic effect button when main camera is active
                const mosaicBtn = document.getElementById('toggle-mosaic-btn');
                if (mosaicBtn) {
                    mosaicBtn.disabled = false;
                }
                
                // Initialize mosaic effect controller if not already done
                if (window.app && window.app.mosaicEffectController && !window.app.mosaicEffectController.videoElement) {
                    window.app.mosaicEffectController.initialize(this.videoElement);
                }
                
                // Apply transforms and color adjustments after camera starts
                this.applyTransforms();
                this.applyColorAdjustments();
            } else {
                this.pipStream = stream;
                if (!this.pipVideo) {
                    this.pipVideo = document.getElementById('pip-video');
                }
                this.pipVideo.srcObject = stream;
                
                // Set PIP video element dimensions to match the actual camera resolution
                this.pipVideo.width = settings.width;
                this.pipVideo.height = settings.height;
                
                updateStatus('pip-status', `PIP Camera: Active (${settings.width}x${settings.height})`);
                
                // Update PIP UI to match saved visibility state
                this.updatePipUI();
            }
            
            // Update toggle button state and text
            const toggleBtnId = type === 'main' ? 'toggle-main-btn' : 'toggle-pip-camera-btn';
            const toggleBtn = document.getElementById(toggleBtnId);
            if (toggleBtn) {
                toggleBtn.textContent = `Stop ${type === 'main' ? 'Main' : 'PiP'} Camera`;
                toggleBtn.classList.add('bg-red-600');
                toggleBtn.classList.remove('bg-blue-600');
            }
        } catch (error) {
            console.error(`Error starting ${type} camera:`, error);
            updateStatus(`${type}-status`, `Error: Could not access camera`);
        }
    }

    /**
     * Stop camera
     * @param {string} type - Camera type ('main' or 'pip')
     */
    stopCamera(type = 'main') {
        if (type === 'main' && this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.videoElement.srcObject = null;
            
            updateStatus('main-status', 'Main Camera: Not Active');
            
            // Disable food detection button when main camera is stopped
            const foodDetectionBtn = document.getElementById('food-detection-btn');
            if (foodDetectionBtn) {
                foodDetectionBtn.disabled = true;
            }
            
            // Disable mosaic effect button when main camera is stopped
            const mosaicBtn = document.getElementById('toggle-mosaic-btn');
            if (mosaicBtn) {
                mosaicBtn.disabled = true;
            }
            
            // Stop mosaic effect if it's running
            if (window.app && window.app.mosaicEffectController && window.app.mosaicEffectController.isEnabled) {
                window.app.mosaicEffectController.stopProcessing();
                window.app.mosaicEffectController.isEnabled = false;
                
                // Reset button state
                const toggleBtn = document.getElementById('toggle-mosaic-btn');
                if (toggleBtn) {
                    toggleBtn.textContent = 'Enable Mosaic';
                    toggleBtn.classList.remove('bg-red-600');
                    toggleBtn.classList.add('bg-purple-600');
                }
                
                // Hide controls
                const mosaicControls = document.getElementById('mosaic-controls');
                if (mosaicControls) {
                    mosaicControls.classList.add('hidden');
                }
            }
            
            // Stop food detection if it's running
            if (window.app && window.app.foodDetectionController && window.app.foodDetectionController.isDetecting) {
                window.app.foodDetectionController.stopFoodDetection();
            }
        } else if (type === 'pip' && this.pipStream) {
            this.pipStream.getTracks().forEach(track => track.stop());
            this.pipStream = null;
            if (this.pipVideo) {
                this.pipVideo.srcObject = null;
            }
            
            updateStatus('pip-status', 'PIP Camera: Not Active');
        }
        
        // Update toggle button state and text
        const toggleBtnId = type === 'main' ? 'toggle-main-btn' : 'toggle-pip-camera-btn';
        const toggleBtn = document.getElementById(toggleBtnId);
        if (toggleBtn) {
            toggleBtn.textContent = `Start ${type === 'main' ? 'Main' : 'PiP'} Camera`;
            toggleBtn.classList.add('bg-blue-600');
            toggleBtn.classList.remove('bg-red-600');
        }
    }

    /**
     * Toggle camera on/off
     * @param {string} type - Camera type ('main' or 'pip')
     */
    toggleCamera(type = 'main') {
        if ((type === 'main' && this.stream) || (type === 'pip' && this.pipStream)) {
            this.stopCamera(type);
        } else {
            this.startCamera(type);
        }
    }

    /**
     * Switch to next available camera
     */
    async switchCamera() {
        if (this.devices.length <= 1) return;
        
        const currentIndex = this.devices.findIndex(device => device.deviceId === this.currentDeviceId);
        const nextIndex = (currentIndex + 1) % this.devices.length;
        this.currentDeviceId = this.devices[nextIndex].deviceId;
        
        await this.startCamera();
        
        // Update camera label
        const cameraLabel = document.getElementById('camera-label');
        if (cameraLabel) {
            cameraLabel.textContent = this.getCameraLabel(this.devices[nextIndex]);
        }
    }

    /**
     * Get camera label
     * @param {MediaDeviceInfo} device - Camera device
     * @returns {string} - Camera label
     */
    getCameraLabel(device) {
        return device.label || `Camera ${this.devices.indexOf(device) + 1}`;
    }

    /**
     * Start video recording
     */
    async startRecording() {
        if (!this.stream) {
            updateStatus('main-status', 'Error: Camera not started');
            return;
        }

        try {
            this.recordedChunks = [];
            
            const options = { mimeType: 'video/webm' };
            
            // Try different MIME types if the first one fails
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = 'video/mp4';
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = 'video/quicktime';
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                        options.mimeType = '';
                    }
                }
            }

            this.mediaRecorder = new MediaRecorder(this.stream, options);
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecordedVideo();
            };
            
            this.mediaRecorder.start();
            this.isRecording = true;
            
            // Update UI
            const recordBtn = document.getElementById('record-video-btn');
            if (recordBtn) {
                updateButton(recordBtn, 'Stop Recording');
                recordBtn.classList.add('bg-red-600');
                recordBtn.classList.remove('bg-blue-600');
            }
            
            updateStatus('main-status', 'Recording started...');
        } catch (error) {
            console.error('Error starting recording:', error);
            updateStatus('main-status', 'Error: Could not start recording');
        }
    }

    /**
     * Stop video recording
     */
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            // Update UI
            const recordBtn = document.getElementById('record-video-btn');
            if (recordBtn) {
                updateButton(recordBtn, 'Record Video');
                recordBtn.classList.remove('bg-red-600');
                recordBtn.classList.add('bg-blue-600');
            }
            
            updateStatus('main-status', 'Recording stopped');
        }
    }

    /**
     * Toggle video recording
     */
    toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            this.startRecording();
        }
    }

    /**
     * Process recorded video
     */
    processRecordedVideo() {
        if (this.recordedChunks.length === 0) return;
        
        const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        
        // Create video element for recorded video
        const recordedVideo = document.createElement('video');
        recordedVideo.src = url;
        recordedVideo.controls = true;
        recordedVideo.className = 'w-full h-full object-cover';
        
        // Replace main video with recorded video
        const videoContainer = document.getElementById('video-container');
        if (videoContainer) {
            videoContainer.innerHTML = '';
            videoContainer.appendChild(recordedVideo);
        }
        
        // Show save button
        const saveBtn = document.getElementById('save-recorded-video-btn');
        if (saveBtn) {
            showElement(saveBtn);
            saveBtn.dataset.blobUrl = url;
        }
        
        updateStatus('main-status', 'Video recorded successfully');
    }

    /**
     * Save recorded video
     */
    saveRecordedVideo() {
        const saveBtn = document.getElementById('save-recorded-video-btn');
        if (!saveBtn || !saveBtn.dataset.blobUrl) return;
        
        const url = saveBtn.dataset.blobUrl;
        const a = document.createElement('a');
        a.href = url;
        a.download = `recording_${new Date().toISOString().replace(/:/g, '-')}.webm`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        
        updateStatus('main-status', 'Video saved successfully');
    }

    /**
     * Create picture-in-picture video
     */
    createPipVideo() {
        if (!this.stream) return;
        
        // Create pip video element
        this.pipVideo = document.createElement('video');
        this.pipVideo.srcObject = this.stream;
        this.pipVideo.autoplay = true;
        this.pipVideo.muted = true;
        this.pipVideo.className = 'w-48 h-36 object-cover rounded-lg shadow-lg';
        
        // Create container
        this.pipVideoContainer = document.createElement('div');
        this.pipVideoContainer.className = 'fixed z-50';
        this.pipVideoContainer.style.left = `${this.pipPosition.x}px`;
        this.pipVideoContainer.style.top = `${this.pipPosition.y}px`;
        this.pipVideoContainer.appendChild(this.pipVideo);
        

        
        // Add to body
        document.body.appendChild(this.pipVideoContainer);
        
        // Setup drag functionality
        this.setupPipDrag();
    }

    /**
     * Setup picture-in-picture drag functionality
     */
    setupPipDrag() {
        if (!this.pipVideoContainer) return;
        
        this.pipVideoContainer.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.dragOffset.x = e.clientX - this.pipPosition.x;
            this.dragOffset.y = e.clientY - this.pipPosition.y;
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!this.isDragging) return;
            
            this.pipPosition.x = e.clientX - this.dragOffset.x;
            this.pipPosition.y = e.clientY - this.dragOffset.y;
            
            this.pipVideoContainer.style.left = `${this.pipPosition.x}px`;
            this.pipVideoContainer.style.top = `${this.pipPosition.y}px`;
        });
        
        document.addEventListener('mouseup', () => {
            this.isDragging = false;
        });
    }

    /**
     * Toggle picture-in-picture visibility
     */
    togglePipVisibility() {
        // Use the existing pip-container element from the template
        this.pipVideoContainer = document.getElementById('pip-container');
        this.pipVideo = document.getElementById('pip-video');
        
        if (!this.pipVideoContainer) {
            console.error('PIP container element not found!');
            return;
        }
        
        // Toggle visibility of the existing PIP container
        const isCurrentlyVisible = this.pipVideoContainer.style.display !== 'none' && 
                                  this.pipVideoContainer.style.visibility !== 'hidden' &&
                                  this.pipVideoContainer.style.opacity !== '0';
        
        if (isCurrentlyVisible) {
            // Hide the PIP container
            this.pipVideoContainer.style.display = 'none';
            this.pipVideoContainer.style.visibility = 'hidden';
            this.pipVideoContainer.style.opacity = '0';
            updateStatus('pip-status', 'PIP Camera: Hidden');
            // Update the pipVisible state
            this.pipVisible = false;
        } else {
            // Show the PIP container
            this.pipVideoContainer.style.display = 'block';
            this.pipVideoContainer.style.visibility = 'visible';
            this.pipVideoContainer.style.opacity = '1';
            updateStatus('pip-status', 'PIP Camera: Visible');
            // Update the pipVisible state
            this.pipVisible = true;
        }
        
        // Save the updated PIP visibility state
        this.saveAllSettings();
    }

    /**
     * Update PIP UI to match the saved visibility state
     */
    updatePipUI() {
        // Get the PIP container element
        const pipContainer = this.pipVideoContainer || document.getElementById('pip-container');
        const togglePipViewBtn = document.getElementById('toggle-pip-visibility-btn');
        
        if (pipContainer) {
            // Set the visibility based on the saved state
            if (this.pipVisible) {
                // Show the PIP container
                pipContainer.style.display = 'block';
                pipContainer.style.visibility = 'visible';
                pipContainer.style.opacity = '1';
                updateStatus('pip-status', 'PIP Camera: Visible');
            } else {
                // Hide the PIP container
                pipContainer.style.display = 'none';
                pipContainer.style.visibility = 'hidden';
                pipContainer.style.opacity = '0';
                updateStatus('pip-status', 'PIP Camera: Hidden');
            }
        }
        
        if (togglePipViewBtn) {
            // Update the button text and icon based on the saved state
            if (this.pipVisible) {
                togglePipViewBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 01-6 0 3 3 0 016 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                    Hide PIP View
                `;
            } else {
                togglePipViewBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Show PIP View
                `;
            }
        }
    }

    /**
     * Toggle PIP view visibility without stopping the stream
     */
    togglePipView() {
        console.log('togglePipView called');
        
        // Use the existing pip-container element from the template
        this.pipVideoContainer = document.getElementById('pip-container');
        this.pipVideo = document.getElementById('pip-video');
        
        if (!this.pipVideoContainer) {
            console.error('PIP container element not found!');
            return;
        }
        
        console.log('Toggling visibility of existing PIP container');
        console.log('Current display:', this.pipVideoContainer.style.display);
        console.log('Current visibility:', this.pipVideoContainer.style.visibility);
        
        // Toggle visibility of the existing PIP container
        const isCurrentlyVisible = this.pipVideoContainer.style.display !== 'none' && 
                                  this.pipVideoContainer.style.visibility !== 'hidden' &&
                                  this.pipVideoContainer.style.opacity !== '0';
        
        if (isCurrentlyVisible) {
            console.log('Hiding PIP container');
            this.pipVideoContainer.style.display = 'none';
            this.pipVideoContainer.style.visibility = 'hidden';
            this.pipVideoContainer.style.opacity = '0';
            
            // Update button text
            const togglePipViewBtn = document.getElementById('toggle-pip-visibility-btn');
            if (togglePipViewBtn) {
                console.log('Updating button text to Show PIP View');
                togglePipViewBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Show PIP View
                `;
                updateStatus('pip-status', 'PIP Camera: Hidden');
            }
            // Update the pipVisible state
            this.pipVisible = false;
        } else {
            console.log('Showing PIP container');
            this.pipVideoContainer.style.display = 'block';
            this.pipVideoContainer.style.visibility = 'visible';
            this.pipVideoContainer.style.opacity = '1';
            
            // Update button text
            const togglePipViewBtn = document.getElementById('toggle-pip-visibility-btn');
            if (togglePipViewBtn) {
                console.log('Updating button text to Hide PIP View');
                togglePipViewBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 01-6 0 3 3 0 016 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                    Hide PIP View
                `;
                updateStatus('pip-status', 'PIP Camera: Visible');
            }
            // Update the pipVisible state
            this.pipVisible = true;
        }
        console.log('togglePipView completed');
        
        // Save the updated PIP visibility state
        this.saveAllSettings();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Request camera permission event
        document.addEventListener('requestPermission', () => this.initializeCameraPermission());
        
        // Check camera permission status on initialization
        this.checkCameraPermission();
        
        // Camera control events from main.js
        document.addEventListener('startCamera', (e) => this.startCamera(e.detail.type));
        document.addEventListener('stopCamera', (e) => this.stopCamera(e.detail.type));
        document.addEventListener('toggleCamera', (e) => this.toggleCamera(e.detail.type));
        document.addEventListener('togglePipVisibility', () => this.togglePipVisibility());
        document.addEventListener('swapCameras', () => this.swapCameras());
        document.addEventListener('movePip', (e) => this.movePip(e.detail.direction));
        
        // Main camera select
        const mainSelect = document.getElementById('main-camera-select');
        if (mainSelect) {
            mainSelect.addEventListener('change', async (e) => {
                if (e.target.value) {
                    // Get capabilities for the selected camera
                    await this.getCameraCapabilities(e.target.value);
                    // Reset resolution selector when camera is changed
                    const resolutionSelect = document.getElementById('resolution-select');
                    if (resolutionSelect) {
                        resolutionSelect.value = '';
                        this.setCameraResolution(null, null);
                    }
                }
            });
        }
        // Camera toggle button
        const cameraBtn = document.getElementById('toggle-camera-btn');
        if (cameraBtn) {
            cameraBtn.addEventListener('click', () => this.toggleCamera());
        }
        
        // Record video button
        const recordBtn = document.getElementById('record-video-btn');
        if (recordBtn) {
            recordBtn.addEventListener('click', () => this.toggleRecording());
        }
        
        // Save recorded video button
        const saveBtn = document.getElementById('save-recorded-video-btn');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveRecordedVideo());
        }
        
        // Switch camera button
        const switchBtn = document.getElementById('switch-camera-btn');
        if (switchBtn) {
            switchBtn.addEventListener('click', () => this.switchCamera());
        }
        
        // Toggle PiP button
        const pipBtn = document.getElementById('toggle-pip-btn');
        if (pipBtn) {
            pipBtn.addEventListener('click', () => this.togglePipVisibility());
        }
        
        // Toggle PIP view visibility button
        const togglePipViewBtn = document.getElementById('toggle-pip-visibility-btn');
        if (togglePipViewBtn) {
            togglePipViewBtn.addEventListener('click', () => {
                console.log('Toggle PIP visibility button clicked');
                this.togglePipView();
            });
        }
        
        // Resolution selector
        const resolutionSelect = document.getElementById('resolution-select');
        if (resolutionSelect) {
            resolutionSelect.addEventListener('change', (e) => {
                const value = e.target.value;
                if (value) {
                    const [width, height] = value.split('x').map(Number);
                    this.setCameraResolution(width, height);
                    // Update the desired resolution property for saving
                    this.desiredResolution = { width, height };
                    updateStatus('main-status', `Resolution set to ${width}x${height}`);
                } else {
                    this.setCameraResolution(null, null);
                    // Reset the desired resolution property
                    this.desiredResolution = null;
                    updateStatus('main-status', 'Resolution reset to default');
                }
            });
        }
        
        // Setup transform controls (rotation and mirror)
        this.setupTransformControls();
        
        // Setup color adjustment controls
        this.setupColorAdjustmentControls();
    }

    /**
     * Check camera permission status
     */
    async checkCameraPermission() {
        try {
            // Check if permission is already granted
            const permissionStatus = await navigator.permissions.query({ name: 'camera' });
            
            if (permissionStatus.state === 'granted') {
                this.hidePermissionPrompt();
                this.loadCameraDevices();
            } else {
                this.showPermissionPrompt();
            }
        } catch (error) {
            console.error('Error checking camera permission:', error);
        }
    }
    
    /**
     * Show permission prompt
     */
    showPermissionPrompt() {
        const permissionPrompt = document.getElementById('permission-prompt');
        if (permissionPrompt) {
            showElement(permissionPrompt);
        }
    }
    
    /**
     * Hide permission prompt
     */
    hidePermissionPrompt() {
        const permissionPrompt = document.getElementById('permission-prompt');
        if (permissionPrompt) {
            hideElement(permissionPrompt);
        }
        
        // Enable camera selection dropdowns and buttons
        const mainCameraSelect = document.getElementById('main-camera-select');
        const pipCameraSelect = document.getElementById('pip-camera-select');
        const toggleMainBtn = document.getElementById('toggle-main-btn');
        const togglePipBtn = document.getElementById('toggle-pip-camera-btn');
        const foodDetectionBtn = document.getElementById('food-detection-btn');
        
        if (mainCameraSelect) mainCameraSelect.disabled = false;
        if (pipCameraSelect) pipCameraSelect.disabled = false;
        if (toggleMainBtn) toggleMainBtn.disabled = false;
        if (togglePipBtn) togglePipBtn.disabled = false;
        if (foodDetectionBtn) foodDetectionBtn.disabled = false;
    }
    
    /**
     * Load camera devices
     */
    async loadCameraDevices() {
        await this.getCameraDevices();
        this.populateCameraSelectors();
    }
    
    /**
     * Populate camera selectors with available devices
     */
    populateCameraSelectors() {
        const mainCameraSelect = document.getElementById('main-camera-select');
        const pipCameraSelect = document.getElementById('pip-camera-select');
        
        if (!mainCameraSelect || !pipCameraSelect) return;
        
        // Clear existing options
        mainCameraSelect.innerHTML = '<option value="">-- Select a camera --</option>';
        pipCameraSelect.innerHTML = '<option value="">-- Select a camera --</option>';
        
        // Add devices to selectors
        this.devices.forEach(device => {
            const mainOption = document.createElement('option');
            mainOption.value = device.deviceId;
            mainOption.textContent = this.getCameraLabel(device);
            mainCameraSelect.appendChild(mainOption);
            
            const pipOption = document.createElement('option');
            pipOption.value = device.deviceId;
            pipOption.textContent = this.getCameraLabel(device);
            pipCameraSelect.appendChild(pipOption);
        });
        
        // After populating, set the saved selections if they exist
        if (this.mainCameraDeviceId && mainCameraSelect.querySelector(`option[value="${this.mainCameraDeviceId}"]`)) {
            mainCameraSelect.value = this.mainCameraDeviceId;
        }
        
        if (this.pipCameraDeviceId && pipCameraSelect.querySelector(`option[value="${this.pipCameraDeviceId}"]`)) {
            pipCameraSelect.value = this.pipCameraDeviceId;
        }
    }
    
    /**
     * Move picture-in-picture window
     * @param {string} direction - Direction to move ('up', 'down', 'left', 'right')
     */
    movePip(direction) {
        if (!this.pipVideoContainer) return;
        
        const step = 10;
        
        switch (direction) {
            case 'up':
                this.pipPosition.y -= step;
                break;
            case 'down':
                this.pipPosition.y += step;
                break;
            case 'left':
                this.pipPosition.x -= step;
                break;
            case 'right':
                this.pipPosition.x += step;
                break;
        }
        
        // Ensure PiP stays within window bounds
        const containerWidth = this.pipVideoContainer.offsetWidth;
        const containerHeight = this.pipVideoContainer.offsetHeight;
        
        this.pipPosition.x = Math.max(0, Math.min(this.pipPosition.x, window.innerWidth - containerWidth));
        this.pipPosition.y = Math.max(0, Math.min(this.pipPosition.y, window.innerHeight - containerHeight));
        
        this.pipVideoContainer.style.left = `${this.pipPosition.x}px`;
        this.pipVideoContainer.style.top = `${this.pipPosition.y}px`;
    }
    
    /**
     * Setup color adjustment controls event listeners
     */
    setupColorAdjustmentControls() {
        // Brightness slider
        const brightnessSlider = document.getElementById('brightness-slider');
        const brightnessValue = document.getElementById('brightness-value');
        if (brightnessSlider && brightnessValue) {
            brightnessSlider.addEventListener('input', (e) => {
                const value = e.target.value;
                brightnessValue.textContent = `${value}%`;
                this.adjustBrightness(value);
            });
        }

        // Contrast slider
        const contrastSlider = document.getElementById('contrast-slider');
        const contrastValue = document.getElementById('contrast-value');
        if (contrastSlider && contrastValue) {
            contrastSlider.addEventListener('input', (e) => {
                const value = e.target.value;
                contrastValue.textContent = `${value}%`;
                this.adjustContrast(value);
            });
        }

        // Saturation slider
        const saturationSlider = document.getElementById('saturation-slider');
        const saturationValue = document.getElementById('saturation-value');
        if (saturationSlider && saturationValue) {
            saturationSlider.addEventListener('input', (e) => {
                const value = e.target.value;
                saturationValue.textContent = `${value}%`;
                this.adjustSaturation(value);
            });
        }

        // Red channel slider
        const redChannelSlider = document.getElementById('red-channel-slider');
        const redChannelValue = document.getElementById('red-channel-value');
        if (redChannelSlider && redChannelValue) {
            redChannelSlider.addEventListener('input', (e) => {
                const value = e.target.value;
                redChannelValue.textContent = `${value}%`;
                this.adjustRedChannel(value);
            });
        }

        // Green channel slider
        const greenChannelSlider = document.getElementById('green-channel-slider');
        const greenChannelValue = document.getElementById('green-channel-value');
        if (greenChannelSlider && greenChannelValue) {
            greenChannelSlider.addEventListener('input', (e) => {
                const value = e.target.value;
                greenChannelValue.textContent = `${value}%`;
                this.adjustGreenChannel(value);
            });
        }

        // Blue channel slider
        const blueChannelSlider = document.getElementById('blue-channel-slider');
        const blueChannelValue = document.getElementById('blue-channel-value');
        if (blueChannelSlider && blueChannelValue) {
            blueChannelSlider.addEventListener('input', (e) => {
                const value = e.target.value;
                blueChannelValue.textContent = `${value}%`;
                this.adjustBlueChannel(value);
            });
        }
    }

    /**
     * Adjust brightness of video element
     * @param {number} value - Brightness percentage (0-200)
     */
    adjustBrightness(value) {
        if (!this.videoElement) return;
        
        // Update the brightness value
        this.brightness = value;
        this.applyColorAdjustments();
    }

    /**
     * Adjust contrast of video element
     * @param {number} value - Contrast percentage (0-200)
     */
    adjustContrast(value) {
        if (!this.videoElement) return;
        
        // Update the contrast value
        this.contrast = value;
        this.applyColorAdjustments();
    }

    /**
     * Adjust saturation of video element
     * @param {number} value - Saturation percentage (0-200)
     */
    adjustSaturation(value) {
        if (!this.videoElement) return;
        
        // Update the saturation value
        this.saturation = value;
        this.applyColorAdjustments();
    }

    /**
     * Adjust red channel of video element
     * @param {number} value - Red channel percentage (0-200)
     */
    adjustRedChannel(value) {
        if (!this.videoElement) return;
        
        // Update the red channel value
        this.redChannel = value;
        this.applyColorAdjustments();
    }

    /**
     * Adjust green channel of video element
     * @param {number} value - Green channel percentage (0-200)
     */
    adjustGreenChannel(value) {
        if (!this.videoElement) return;
        
        // Update the green channel value
        this.greenChannel = value;
        this.applyColorAdjustments();
    }

    /**
     * Adjust blue channel of video element
     * @param {number} value - Blue channel percentage (0-200)
     */
    adjustBlueChannel(value) {
        if (!this.videoElement) return;
        
        // Update the blue channel value
        this.blueChannel = value;
        this.applyColorAdjustments();
    }

    /**
     * Apply all color adjustments to video element using CSS filters
     */
    applyColorAdjustments() {
        if (!this.videoElement) return;
        
        // Set default values if not defined
        this.brightness = this.brightness || 100;
        this.contrast = this.contrast || 100;
        this.saturation = this.saturation || 100;
        this.redChannel = this.redChannel || 100;
        this.greenChannel = this.greenChannel || 100;
        this.blueChannel = this.blueChannel || 100;

        // Since CSS doesn't support individual RGB channel adjustments directly,
        // we'll use a combination of available CSS filters to approximate the effect
        // For red channel issues, reducing red and increasing blue/green can help
        
        // Calculate adjusted values based on channel adjustments
        // Note: We can't directly control individual RGB channels with CSS filters,
        // but we can achieve some color balance adjustment through combinations of filters
        let filterString = '';
        
        // Apply basic adjustments
        if (this.brightness !== 100) {
            filterString += `brightness(${this.brightness}%) `;
        }
        if (this.contrast !== 100) {
            filterString += `contrast(${this.contrast}%) `;
        }
        if (this.saturation !== 100) {
            filterString += `saturate(${this.saturation}%) `;
        }
        
        // To address the red skew, we can use hue-rotate to shift colors
        // and opacity adjustments to simulate channel balancing
        // Calculate hue rotation based on red excess (reducing red dominance)
        const redExcess = this.redChannel - 100;
        const blueDeficit = 100 - this.blueChannel;
        
        // If red channel is too high, apply a slight hue rotation toward cyan
        if (redExcess > 0 || blueDeficit > 0) {
            // Calculate an appropriate hue rotation value to compensate for red excess
            const hueAdjustment = Math.min(20, Math.max(-20, (blueDeficit - redExcess) / 2));
            if (Math.abs(hueAdjustment) > 0.5) {  // Only apply if significant
                filterString += `hue-rotate(${hueAdjustment}deg) `;
            }
        }
        
        // Apply the combined CSS filters to the video element
        this.videoElement.style.filter = filterString.trim();
    }
    
    /**
     * Save all camera settings to local storage
     */
    saveAllSettings() {
        const settings = {
            // Color adjustments
            brightness: this.brightness,
            contrast: this.contrast,
            saturation: this.saturation,
            redChannel: this.redChannel,
            greenChannel: this.greenChannel,
            blueChannel: this.blueChannel,
            
            // Transform settings
            rotationAngle: this.rotationAngle,
            isMirrored: this.isMirrored,
            
            // PIP settings
            pipVisible: this.pipVisible,
            
            // Camera selections
            mainCameraDeviceId: this.mainCameraDeviceId,
            pipCameraDeviceId: this.pipCameraDeviceId,
            currentDeviceId: this.currentDeviceId,
            
            // Resolution settings - save as string format "WIDTHxHEIGHT" to match UI
            desiredResolution: this.desiredResolution ? `${this.desiredResolution.width}x${this.desiredResolution.height}` : null
        };
        
        localStorage.setItem('cameraSettings', JSON.stringify(settings));
    }
    
    /**
     * Load all camera settings from local storage
     */
    loadAllSettings() {
        try {
            const savedSettings = localStorage.getItem('cameraSettings');
            if (savedSettings) {
                const settings = JSON.parse(savedSettings);
                
                // Load color adjustments
                this.brightness = settings.brightness !== undefined ? settings.brightness : 100;
                this.contrast = settings.contrast !== undefined ? settings.contrast : 100;
                this.saturation = settings.saturation !== undefined ? settings.saturation : 100;
                this.redChannel = settings.redChannel !== undefined ? settings.redChannel : 100;
                this.greenChannel = settings.greenChannel !== undefined ? settings.greenChannel : 100;
                this.blueChannel = settings.blueChannel !== undefined ? settings.blueChannel : 100;
                
                // Load transform settings
                this.rotationAngle = settings.rotationAngle !== undefined ? settings.rotationAngle : 0;
                this.isMirrored = settings.isMirrored !== undefined ? settings.isMirrored : false;
                
                // Load PIP settings
                this.pipVisible = settings.pipVisible !== undefined ? settings.pipVisible : false;
                
                // Load camera selections
                this.mainCameraDeviceId = settings.mainCameraDeviceId !== undefined ? settings.mainCameraDeviceId : null;
                this.pipCameraDeviceId = settings.pipCameraDeviceId !== undefined ? settings.pipCameraDeviceId : null;
                this.currentDeviceId = settings.currentDeviceId !== undefined ? settings.currentDeviceId : null;
                
                // Load resolution settings - convert string format "WIDTHxHEIGHT" back to object format
                if (settings.desiredResolution) {
                    if (typeof settings.desiredResolution === 'string') {
                        // Parse string format like "1920x1080"
                        const [widthStr, heightStr] = settings.desiredResolution.split('x');
                        if (widthStr && heightStr) {
                            this.desiredResolution = {
                                width: parseInt(widthStr),
                                height: parseInt(heightStr)
                            };
                        } else {
                            this.desiredResolution = null;
                        }
                    } else {
                        // Already in object format
                        this.desiredResolution = settings.desiredResolution;
                    }
                } else {
                    this.desiredResolution = null;
                }
                
                // Update the UI sliders to reflect loaded values
                this.updateColorSliderValues();
                
                // Update UI controls to reflect loaded transform settings
                this.updateTransformUI();
                
                // Apply the loaded transforms if video element is ready
                if (this.videoElement && this.stream) {
                    this.applyTransforms();
                    // Use a slight delay to ensure the video element is properly initialized before applying color adjustments
                    setTimeout(() => {
                        this.applyColorAdjustments();
                    }, 100);
                }
                
                // Update resolution selection in UI if resolution was loaded
                if (this.desiredResolution) {
                    const resolutionSelect = document.getElementById('resolution-select');
                    if (resolutionSelect && resolutionSelect.querySelector(`option[value="${this.desiredResolution.width}x${this.desiredResolution.height}"]`)) {
                        resolutionSelect.value = `${this.desiredResolution.width}x${this.desiredResolution.height}`;
                    }
                    
                    // If camera is already running, apply the resolution
                    if (this.videoElement && this.stream) {
                        this.setCameraResolution(this.desiredResolution.width, this.desiredResolution.height);
                    }
                }
                
                // Apply the loaded PIP visibility state by ensuring the container visibility matches the saved state
                // Since the PIP container is in the HTML template, we need to show/hide it according to the saved state
                setTimeout(() => {
                    // Wait a bit for the DOM to be ready, then ensure the visibility matches the saved state
                    if (this.pipVideoContainer || document.getElementById('pip-container')) {
                        // Update the UI to match the saved state
                        this.updatePipUI();
                    }
                    
                    // Update camera selection dropdowns to reflect loaded values after DOM is ready
                    this.updateCameraSelectionUI();
                }, 100); // Small delay to ensure UI is ready
                
                return true;
            }
        } catch (error) {
            console.error('Error loading camera settings from local storage:', error);
        }
        
        return false;
    }
    
    /**
     * Update UI controls to reflect current transform settings
     */
    updateTransformUI() {
        // Update rotation select dropdown
        const rotationSelect = document.getElementById('rotation-select');
        if (rotationSelect) {
            rotationSelect.value = this.rotationAngle;
        }
        
        // Update mirror toggle button
        const mirrorToggle = document.getElementById('mirror-toggle');
        const mirrorStatus = document.getElementById('mirror-status');
        
        if (mirrorStatus) {
            mirrorStatus.textContent = this.isMirrored ? 'On' : 'Off';
        }
        
        if (mirrorToggle) {
            mirrorToggle.classList.toggle('bg-gray-600', !this.isMirrored);
            mirrorToggle.classList.toggle('bg-blue-600', this.isMirrored);
        }
    }
    
    /**
     * Load color adjustments from local storage
     */
    loadColorAdjustments() {
        try {
            const savedAdjustments = localStorage.getItem('cameraColorAdjustments');
            if (savedAdjustments) {
                const adjustments = JSON.parse(savedAdjustments);
                
                this.brightness = adjustments.brightness !== undefined ? adjustments.brightness : 100;
                this.contrast = adjustments.contrast !== undefined ? adjustments.contrast : 100;
                this.saturation = adjustments.saturation !== undefined ? adjustments.saturation : 100;
                this.redChannel = adjustments.redChannel !== undefined ? adjustments.redChannel : 100;
                this.greenChannel = adjustments.greenChannel !== undefined ? adjustments.greenChannel : 100;
                this.blueChannel = adjustments.blueChannel !== undefined ? adjustments.blueChannel : 100;
                
                // Update the UI sliders to reflect loaded values
                this.updateColorSliderValues();
                
                return true;
            }
        } catch (error) {
            console.error('Error loading color adjustments from local storage:', error);
        }
        
        return false;
    }
    
    /**
     * Update the color slider values in the UI to match stored values
     */
    updateColorSliderValues() {
        const brightnessSlider = document.getElementById('brightness-slider');
        const contrastSlider = document.getElementById('contrast-slider');
        const saturationSlider = document.getElementById('saturation-slider');
        const redChannelSlider = document.getElementById('red-channel-slider');
        const greenChannelSlider = document.getElementById('green-channel-slider');
        const blueChannelSlider = document.getElementById('blue-channel-slider');
        
        const brightnessValueDisplay = document.getElementById('brightness-value');
        const contrastValueDisplay = document.getElementById('contrast-value');
        const saturationValueDisplay = document.getElementById('saturation-value');
        const redValueDisplay = document.getElementById('red-channel-value');
        const greenValueDisplay = document.getElementById('green-channel-value');
        const blueValueDisplay = document.getElementById('blue-channel-value');
        
        if (brightnessSlider) brightnessSlider.value = this.brightness;
        if (contrastSlider) contrastSlider.value = this.contrast;
        if (saturationSlider) saturationSlider.value = this.saturation;
        if (redChannelSlider) redChannelSlider.value = this.redChannel;
        if (greenChannelSlider) greenChannelSlider.value = this.greenChannel;
        if (blueChannelSlider) blueChannelSlider.value = this.blueChannel;
        
        if (brightnessValueDisplay) brightnessValueDisplay.textContent = `${this.brightness}%`;
        if (contrastValueDisplay) contrastValueDisplay.textContent = `${this.contrast}%`;
        if (saturationValueDisplay) saturationValueDisplay.textContent = `${this.saturation}%`;
        if (redValueDisplay) redValueDisplay.textContent = `${this.redChannel}%`;
        if (greenValueDisplay) greenValueDisplay.textContent = `${this.greenChannel}%`;
        if (blueValueDisplay) blueValueDisplay.textContent = `${this.blueChannel}%`;
    }

    /**
     * Update camera selection UI to reflect loaded values
     */
    updateCameraSelectionUI() {
        // Update main camera selection dropdown
        const mainCameraSelect = document.getElementById('main-camera-select');
        if (mainCameraSelect && this.mainCameraDeviceId) {
            // Wait for options to be populated, then set the value
            setTimeout(() => {
                mainCameraSelect.value = this.mainCameraDeviceId;
            }, 100);
        }
        
        // Update PIP camera selection dropdown
        const pipCameraSelect = document.getElementById('pip-camera-select');
        if (pipCameraSelect && this.pipCameraDeviceId) {
            // Wait for options to be populated, then set the value
            setTimeout(() => {
                pipCameraSelect.value = this.pipCameraDeviceId;
            }, 100);
        }
        
        // Update resolution selection dropdown
        const resolutionSelect = document.getElementById('resolution-select');
        if (resolutionSelect && this.desiredResolution) {
            // Wait for options to be populated, then set the value
            setTimeout(() => {
                resolutionSelect.value = `${this.desiredResolution.width}x${this.desiredResolution.height}`;
            }, 100);
        }
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        this.stopCamera('main');
        this.stopCamera('pip');
        
        if (this.pipVideoContainer && this.pipVideoContainer.parentNode) {
            this.pipVideoContainer.parentNode.removeChild(this.pipVideoContainer);
        }
        

    }
}
