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
        this.pipVideo = null;
        this.pipVideoContainer = null;
        this.pipPosition = { x: 10, y: 10 };
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
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
        await this.getCameraDevices();
        await this.startCamera();
        this.setupEventListeners();
    }

    /**
     * Get available camera devices
     */
    async getCameraDevices() {
        try {
            const allDevices = await navigator.mediaDevices.enumerateDevices();
            this.devices = allDevices.filter(device => device.kind === 'videoinput');
            
            if (this.devices.length > 0) {
                this.currentDeviceId = this.devices[0].deviceId;
            }
        } catch (error) {
            console.error('Error getting camera devices:', error);
            updateStatus('main-status', 'Error: Could not access camera devices');
        }
    }

    /**
     * Start camera with current device
     * @param {string} type - Camera type ('main' or 'pip')
     */
    async startCamera(type = 'main') {
        try {
            // Get selected device ID
            const selectElement = document.getElementById(type === 'main' ? 'main-camera-select' : 'pip-camera-select');
            if (!selectElement || !selectElement.value) {
                updateStatus(`${type}-status`, `Error: No camera selected`);
                return;
            }
            
            const deviceId = selectElement.value;
            
            // Stop existing stream if it exists
            if (type === 'main' && this.stream) {
                this.stream.getTracks().forEach(track => track.stop());
            } else if (type === 'pip' && this.pipStream) {
                this.pipStream.getTracks().forEach(track => track.stop());
            }

            const constraints = {
                video: { deviceId: { exact: deviceId } }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            
            // Assign stream to the appropriate video element
            if (type === 'main') {
                this.stream = stream;
                this.videoElement.srcObject = stream;
                updateStatus('main-status', 'Main Camera: Active');
                
                // Enable food detection button when main camera is active
                const foodDetectionBtn = document.getElementById('food-detection-btn');
                if (foodDetectionBtn) {
                    foodDetectionBtn.disabled = false;
                }
            } else {
                this.pipStream = stream;
                if (!this.pipVideo) {
                    this.pipVideo = document.getElementById('pip-video');
                }
                this.pipVideo.srcObject = stream;
                updateStatus('pip-status', 'PIP Camera: Active');
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
        if (!this.pipVideoContainer) {
            this.createPipVideo();
        } else {
            if (this.pipVideoContainer.parentNode) {
                this.pipVideoContainer.parentNode.removeChild(this.pipVideoContainer);
                this.pipVideoContainer = null;
                this.pipVideo = null;
            } else {
                document.body.appendChild(this.pipVideoContainer);
            }
        }
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
