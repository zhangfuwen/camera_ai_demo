/**
 * Video Recorder Controller
 * Handles recording video from #main-video element every 10 seconds and uploading to server
 */
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
    }

    /**
     * Initialize the video recorder controller
     */
    initialize() {
        console.log('Initializing Video Recorder Controller...');
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

        document.addEventListener('saveRecordedVideo', () => {
            this.saveRecordedVideo();
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
            const stream = videoElement.srcObject;
            
            // Create MediaRecorder instance
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'video/mp4'
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

            console.log('Video recording started');
            this.updateUI();
            this.updateStatus('Video recording: Active');

            // Set up interval to process recordings every 10 seconds
            this.recordingInterval = setInterval(() => {
                this.processCurrentRecording();
            }, this.recordingDuration);

        } catch (error) {
            console.error('Error starting video recording:', error);
            this.updateStatus('Error starting recording: ' + error.message);
        }
    }

    /**
     * Stop video recording
     */
    stopRecording() {
        console.log('Stopping video recording...');
        
        // Clear the recording interval first
        if (this.recordingInterval) {
            clearInterval(this.recordingInterval);
            this.recordingInterval = null;
        }

        // Stop the MediaRecorder but NOT the stream
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            console.log('Stopping MediaRecorder...');
            this.mediaRecorder.stop();
            // Don't touch the stream - let it continue for the main video
        }

        this.isRecording = false;
        console.log('Video recording stopped - main camera stream should continue');
        
        // Verify that the main video stream is still active
        setTimeout(() => {
            if (this.verifyMainVideoStream()) {
                console.log('✅ Main camera stream confirmed active after recording stop');
                this.addUploadLog('Recording stopped - camera stream active', 'success');
            } else {
                console.error('❌ Main camera stream may have been affected');
                this.addUploadLog('⚠️ Recording stopped - camera stream issue detected', 'error');
            }
        }, 500); // Small delay to allow stream to stabilize
        
        this.updateUI();
        this.updateStatus('Video recording: Stopped');
    }

    /**
     * Process current recording chunk and start a new one
     */
    processCurrentRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            console.log('Processing current recording chunk...');
            
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
            console.error('No video stream available for new recording chunk');
            return;
        }

        // Get the existing stream - don't create a new one
        const stream = videoElement.srcObject;
        console.log('Starting new recording chunk from existing stream');
        
        // Create new MediaRecorder instance using the same stream
        this.mediaRecorder = new MediaRecorder(stream, {
            mimeType: 'video/mp4'
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
        console.log('New recording chunk started');
    }

    /**
     * Process the recorded video chunks
     */
    processRecording() {
        if (this.recordedChunks.length === 0) {
            return;
        }

        // Create blob from recorded chunks
        const blob = new Blob(this.recordedChunks, { type: 'video/mp4' });
        
        // Create timestamp for filename
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `video_${timestamp}.mp4`;

        // Add to upload queue
        this.uploadQueue.push({
            blob: blob,
            filename: filename,
            timestamp: timestamp
        });

        console.log(`Video chunk recorded: ${filename}`);
        
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
        // Find the video recorder section
        const videoRecorderSection = document.querySelector('#record-video-btn').closest('.accordion-content');
        if (!videoRecorderSection) {
            console.error('Could not find video recorder section for log container');
            return null;
        }

        // Create log container
        const logContainer = document.createElement('div');
        logContainer.id = 'upload-log-container';
        logContainer.className = 'mt-4 p-3 bg-gray-900 rounded border border-gray-700 max-h-48 overflow-y-auto';
        
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
     * Save recorded video locally
     */
    async saveRecordedVideo() {
        if (this.recordedChunks.length === 0) {
            this.updateStatus('No recorded video to save');
            return;
        }

        try {
            const blob = new Blob(this.recordedChunks, { type: 'video/mp4' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `recorded_video_${new Date().toISOString().replace(/[:.]/g, '-')}.mp4`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.updateStatus('Video saved locally');
        } catch (error) {
            console.error('Error saving video:', error);
            this.updateStatus('Error saving video: ' + error.message);
        }
    }

    /**
     * Update UI elements
     */
    updateUI() {
        const recordBtn = document.getElementById('record-video-btn');
        const saveBtn = document.getElementById('save-recorded-video-btn');
        const statusDiv = document.getElementById('recording-status');

        if (recordBtn) {
            if (this.isRecording) {
                recordBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <rect x="6" y="6" width="12" height="12" stroke-width="2"/>
                    </svg>
                    Stop Recording
                `;
                recordBtn.classList.remove('bg-blue-600', 'hover:bg-blue-700');
                recordBtn.classList.add('bg-red-600', 'hover:bg-red-700');
            } else {
                recordBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    Start Recording
                `;
                recordBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
                recordBtn.classList.add('bg-blue-600', 'hover:bg-blue-700');
            }
            
            recordBtn.disabled = false;
        }

        if (saveBtn) {
            saveBtn.disabled = this.recordedChunks.length === 0;
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
