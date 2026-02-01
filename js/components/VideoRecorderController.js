/**
 * Video Recorder Controller
 * Handles recording video from #main-video element every 10 seconds and uploading to server
 */
import { Logger } from '../utils/logger.js';

export class VideoRecorderController {
    constructor() {
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.recordingInterval = null;
        this.isRecording = false;
        this.recordingStartTime = null;
        this.uploadQueue = [];
        this.isUploading = false;
        this.recordingDuration = 10000; // 10 seconds
        this.uploadEndpoint = '/upload_video';
        this.videoStream = null;
        this.logger = new Logger('VideoRecorderController', 'INFO');
    }

    /**
     * Initialize the video recorder controller
     */
    initialize() {
        this.logger.info('Initializing Video Recorder Controller...');
        this.setupEventListeners();
        this.updateUI();
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Listen for custom events
        document.addEventListener('toggleVideoRecording', () => {
            this.toggleRecording();
        });
    }

    /**
     * Toggle video recording on/off
     */
    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }

    /**
     * Start video recording
     */
    async startRecording() {
        try {
            const videoElement = document.getElementById('main-video');
            if (!videoElement || !videoElement.srcObject) {
                throw new Error('No video stream available for recording');
            }

            // Get the video stream
            this.videoStream = videoElement.srcObject;
            
            this.logger.debug('Starting video recording from main-video element');
            
            // Create MediaRecorder instance
            this.mediaRecorder = new MediaRecorder(this.videoStream, {
                mimeType: 'video/webm;codecs=vp9'
            });

            // Set up event handlers
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.processRecording();
            };

            // Start recording
            this.recordedChunks = [];
            this.mediaRecorder.start();
            this.recordingStartTime = Date.now();
            this.isRecording = true;

            this.logger.info('Video recording started');
            this.updateUI();
            this.updateStatus('Video recording: Active');
            this.addUploadLog('Video recording started', 'success');

            // Set up interval to process recordings every 10 seconds
            this.recordingInterval = setInterval(() => {
                this.processCurrentRecording();
            }, this.recordingDuration);

        } catch (error) {
            this.logger.error('Error starting video recording:', error);
            this.updateStatus('Error starting recording: ' + error.message);
            this.addUploadLog(`❌ Error starting recording: ${error.message}`, 'error');
        }
    }

    /**
     * Stop video recording
     */
    stopRecording() {
        this.logger.info('Stopping video recording...');
        
        // Clear the recording interval first
        if (this.recordingInterval) {
            clearInterval(this.recordingInterval);
            this.recordingInterval = null;
        }

        // Stop the MediaRecorder but NOT the stream
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.logger.debug('Stopping MediaRecorder...');
            this.mediaRecorder.stop();
            // Don't touch the stream - let it continue for the main video
        }

        this.isRecording = false;
        this.logger.info('Video recording stopped - main camera stream should continue');
        
        // Verify that the main video stream is still active
        setTimeout(() => {
            if (this.verifyMainVideoStream()) {
                this.logger.info('Main camera stream confirmed active after recording stop');
                this.addUploadLog('Recording stopped - camera stream active', 'success');
            } else {
                this.logger.error('Main camera stream may have been affected');
                this.addUploadLog('⚠️ Recording stopped - camera stream issue detected', 'error');
            }
        }, 500); // Small delay to allow stream to stabilize
        
        this.updateUI();
        this.updateStatus('Video recording: Stopped');
        this.addUploadLog('Video recording stopped', 'info');
    }

    /**
     * Process current recording chunk and start a new one
     */
    processCurrentRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.logger.debug('Processing current recording chunk...');
            
            // Stop current MediaRecorder only (not the stream)
            this.mediaRecorder.stop();
            
            // Start a new recording immediately without affecting the main video
            setTimeout(() => {
                if (this.isRecording) {
                    this.startNewRecordingChunk();
                }
            }, 100); // Small delay to ensure clean transition
        }
    }

    /**
     * Start a new recording chunk
     */
    startNewRecordingChunk() {
        const videoElement = document.getElementById('main-video');
        if (!videoElement || !videoElement.srcObject) {
            this.logger.error('No video stream available for new recording chunk');
            return;
        }

        // Get the existing stream - don't create a new one
        const stream = videoElement.srcObject;
        this.logger.debug('Starting new recording chunk from existing stream');
        
        // Create new MediaRecorder instance using the same stream
        this.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'video/webm;codecs=vp9'
        });

        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.recordedChunks.push(event.data);
            }
        };

        this.mediaRecorder.onstop = () => {
            this.processRecording();
        };

        this.recordedChunks = [];
        this.mediaRecorder.start();
        this.recordingStartTime = Date.now();
        this.logger.debug('New recording chunk started');
    }

    /**
     * Process the recorded video chunks
     */
    processRecording() {
        if (this.recordedChunks.length === 0) {
            return;
        }

        // Create blob from recorded chunks
        const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
        
        // Create timestamp for filename
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `video_${timestamp}.webm`;

        // Add to upload queue
        this.uploadQueue.push({
            blob: blob,
            filename: filename,
            timestamp: timestamp
        });

        this.logger.debug(`Video chunk recorded: ${filename}`);
        this.addUploadLog(`Video recorded: ${filename} (${(blob.size / 1024 / 1024).toFixed(1)} MB)`, 'info');
        
        // Process upload queue
        this.processUploadQueue();
    }

    /**
     * Process the upload queue
     */
    async processUploadQueue() {
        if (this.isUploading || this.uploadQueue.length === 0) {
            return;
        }

        this.isUploading = true;
        
        while (this.uploadQueue.length > 0) {
            const videoData = this.uploadQueue.shift();
            try {
                await this.uploadVideo(videoData);
                console.log(`Successfully uploaded: ${videoData.filename}`);
            } catch (error) {
                console.error(`Failed to upload ${videoData.filename}:`, error);
                // Re-add to queue for retry
                this.uploadQueue.unshift(videoData);
                break;
            }
        }
        
        this.isUploading = false;
    }

    /**
     * Upload video to server
     */
    async uploadVideo(videoData) {
        const formData = new FormData();
        formData.append('video', videoData.blob, videoData.filename);
        formData.append('timestamp', videoData.timestamp);

        // Log upload attempt
        console.log(`[VIDEO UPLOAD] Starting upload: ${videoData.filename}`);
        console.log(`[VIDEO UPLOAD] File size: ${(videoData.blob.size / 1024 / 1024).toFixed(2)} MB`);
        console.log(`[VIDEO UPLOAD] Timestamp: ${videoData.timestamp}`);
        
        // Add log to index.html
        this.addUploadLog(`Uploading: ${videoData.filename} (${(videoData.blob.size / 1024 / 1024).toFixed(2)} MB)`);

        try {
            const startTime = Date.now();
            const response = await fetch(this.uploadEndpoint, {
                method: 'POST',
                body: formData
            });

            const uploadTime = Date.now() - startTime;

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`[VIDEO UPLOAD] Upload failed: ${response.statusText} - ${errorText}`);
                this.addUploadLog(`❌ Upload failed: ${videoData.filename} - ${response.statusText}`, 'error');
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const result = await response.json();
            console.log(`[VIDEO UPLOAD] Upload successful: ${videoData.filename}`);
            console.log(`[VIDEO UPLOAD] Server response:`, result);
            console.log(`[VIDEO UPLOAD] Upload time: ${uploadTime}ms`);
            
            // Add success log to index.html
            this.addUploadLog(`✅ Upload successful: ${videoData.filename} (${uploadTime}ms)`, 'success');

            return result;
        } catch (error) {
            console.error(`[VIDEO UPLOAD] Error uploading ${videoData.filename}:`, error);
            this.addUploadLog(`❌ Upload error: ${videoData.filename} - ${error.message}`, 'error');
            throw error;
        }
    }

    /**
     * Verify that the main video stream is still active
     */
    verifyMainVideoStream() {
        const videoElement = document.getElementById('main-video');
        if (!videoElement) {
            console.error('Main video element not found');
            return false;
        }

        if (!videoElement.srcObject) {
            console.error('Main video stream is null');
            return false;
        }

        // Check if stream is still active
        const stream = videoElement.srcObject;
        const videoTracks = stream.getVideoTracks();
        
        if (videoTracks.length === 0) {
            console.error('No video tracks found in main stream');
            return false;
        }

        const videoTrack = videoTracks[0];
        if (videoTrack.readyState !== 'live') {
            console.error('Main video track is not live:', videoTrack.readyState);
            return false;
        }

        console.log('Main video stream verified - still active and live');
        return true;
    }

    /**
     * Add upload log entry to the UI
     */
    addUploadLog(message, type = 'info') {
        // Create or find the upload log container
        let logContainer = document.getElementById('upload-log-container');
        if (!logContainer) {
            logContainer = this.createUploadLogContainer();
        }

        // Create log entry
        const logEntry = document.createElement('div');
        const timestamp = new Date().toLocaleTimeString();
        logEntry.className = `upload-log-entry upload-log-${type}`;
        logEntry.innerHTML = `<span class="log-time">[${timestamp}]</span> ${message}`;

        // Add to container (prepend for newest first)
        logContainer.insertBefore(logEntry, logContainer.firstChild);

        // Limit log entries to prevent memory issues
        const maxEntries = 50;
        while (logContainer.children.length > maxEntries) {
            logContainer.removeChild(logContainer.lastChild);
        }

        // Auto-scroll to top
        logContainer.scrollTop = 0;
    }

    /**
     * Create upload log container in the UI
     */
    createUploadLogContainer() {
        // Find the video recorder section - look for the parent div with the video recorder content
        const videoRecorderSection = document.querySelector('#record-video-btn').closest('.mb-6');
        if (!videoRecorderSection) {
            this.logger.error('Could not find video recorder section for log container');
            return null;
        }

        // Create log container
        const logContainer = document.createElement('div');
        logContainer.id = 'upload-log-container';
        logContainer.className = 'mt-4 p-3 bg-gray-900 rounded border border-gray-700 max-h-48 overflow-y-auto hidden';
        
        // Add title
        const title = document.createElement('div');
        title.className = 'text-sm font-bold text-gray-300 mb-2';
        title.textContent = 'Upload Log:';
        logContainer.appendChild(title);

        // Insert before the recording status
        const statusDiv = document.getElementById('recording-status');
        if (statusDiv) {
            statusDiv.parentNode.insertBefore(logContainer, statusDiv.nextSibling);
        } else {
            videoRecorderSection.appendChild(logContainer);
        }

        // Add styles for different log types
        const style = document.createElement('style');
        style.textContent = `
            .upload-log-entry {
                font-size: 12px;
                padding: 2px 0;
                border-bottom: 1px solid #374151;
            }
            .upload-log-entry:last-child {
                border-bottom: none;
            }
            .log-time {
                color: #9CA3AF;
                font-weight: bold;
            }
            .upload-log-success {
                color: #10B981;
            }
            .upload-log-error {
                color: #EF4444;
            }
            .upload-log-info {
                color: #3B82F6;
            }
        `;
        document.head.appendChild(style);

        return logContainer;
    }

    /**
     * Update UI elements
     */
    updateUI() {
        const recordBtn = document.getElementById('record-video-btn');
        const statusDiv = document.getElementById('recording-status');

        if (recordBtn) {
            if (this.isRecording) {
                recordBtn.innerHTML = `
                    Stop Recording
                `;
                recordBtn.classList.remove('bg-blue-600', 'hover:bg-blue-700');
                recordBtn.classList.add('bg-red-600', 'hover:bg-red-700');
            } else {
                recordBtn.innerHTML = `
                    Start Recording
                `;
                recordBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
                recordBtn.classList.add('bg-blue-600', 'hover:bg-blue-700');
            }
            
            recordBtn.disabled = false;
        }

        if (statusDiv) {
            const queueInfo = this.uploadQueue.length > 0 ? ` (Queue: ${this.uploadQueue.length})` : '';
            statusDiv.textContent = `Recording: ${this.isRecording ? 'Active' : 'Not Active'}${queueInfo}`;
        }
    }

    /**
     * Update status display
     */
    updateStatus(message) {
        const statusElement = document.getElementById('main-status');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }

    /**
     * Get recording statistics
     */
    getRecordingStats() {
        return {
            isRecording: this.isRecording,
            queueLength: this.uploadQueue.length,
            isUploading: this.isUploading,
            recordingDuration: this.recordingDuration
        };
    }
}
