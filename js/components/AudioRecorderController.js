/**
 * Audio Recorder Controller
 * Handles recording audio every 10 seconds and uploading to server
 */
export class AudioRecorderController {
    constructor() {
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.recordingInterval = null;
        this.isRecording = false;
        this.recordingStartTime = null;
        this.uploadQueue = [];
        this.isUploading = false;
        this.recordingDuration = 10000; // 10 seconds
        this.uploadEndpoint = '/upload_audio';
        this.audioStream = null;
    }

    /**
     * Initialize the audio recorder controller
     */
    initialize() {
        console.log('Initializing Audio Recorder Controller...');
        this.setupEventListeners();
        this.updateUI();
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Listen for custom events
        document.addEventListener('toggleAudioRecording', () => {
            this.toggleRecording();
        });

        document.addEventListener('saveRecordedAudio', () => {
            this.saveRecordedAudio();
        });
    }

    /**
     * Toggle audio recording on/off
     */
    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }

    /**
     * Start audio recording
     */
    async startRecording() {
        try {
            // Request microphone access
            console.log('Requesting microphone access...');
            this.audioStream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 44100
                } 
            });
            
            console.log('Microphone access granted');
            
            // Create MediaRecorder instance
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: 'audio/webm;codecs=opus'
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

            console.log('Audio recording started');
            this.updateUI();
            this.updateStatus('Audio recording: Active');
            this.addUploadLog('Audio recording started', 'success');

            // Set up interval to process recordings every 10 seconds
            this.recordingInterval = setInterval(() => {
                this.processCurrentRecording();
            }, this.recordingDuration);

        } catch (error) {
            console.error('Error starting audio recording:', error);
            this.updateStatus('Error starting audio recording: ' + error.message);
            this.addUploadLog(`❌ Error starting recording: ${error.message}`, 'error');
        }
    }

    /**
     * Stop audio recording
     */
    stopRecording() {
        console.log('Stopping audio recording...');
        
        // Clear the recording interval first
        if (this.recordingInterval) {
            clearInterval(this.recordingInterval);
            this.recordingInterval = null;
        }

        // Stop the MediaRecorder
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            console.log('Stopping Audio MediaRecorder...');
            this.mediaRecorder.stop();
        }

        // Stop the audio stream
        if (this.audioStream) {
            console.log('Stopping audio stream...');
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
        }

        this.isRecording = false;
        console.log('Audio recording stopped');
        
        this.updateUI();
        this.updateStatus('Audio recording: Stopped');
        this.addUploadLog('Audio recording stopped', 'info');
    }

    /**
     * Process current recording chunk and start a new one
     */
    processCurrentRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            console.log('Processing current audio recording chunk...');
            
            // Stop current MediaRecorder
            this.mediaRecorder.stop();
            
            // Start a new recording immediately
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
    async startNewRecordingChunk() {
        try {
            // Create new MediaRecorder instance using the same stream
            this.mediaRecorder = new MediaRecorder(this.audioStream, {
                mimeType: 'audio/webm;codecs=opus'
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
            console.log('New audio recording chunk started');
        } catch (error) {
            console.error('Error starting new audio recording chunk:', error);
            this.addUploadLog(`❌ Error starting new chunk: ${error.message}`, 'error');
        }
    }

    /**
     * Process the recorded audio chunks
     */
    processRecording() {
        if (this.recordedChunks.length === 0) {
            return;
        }

        // Create blob from recorded chunks
        const blob = new Blob(this.recordedChunks, { type: 'audio/webm' });
        
        // Create timestamp for filename
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `audio_${timestamp}.webm`;

        // Add to upload queue
        this.uploadQueue.push({
            blob: blob,
            filename: filename,
            timestamp: timestamp
        });

        console.log(`Audio chunk recorded: ${filename}`);
        this.addUploadLog(`Audio recorded: ${filename} (${(blob.size / 1024).toFixed(1)} KB)`, 'info');
        
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
            const audioData = this.uploadQueue.shift();
            try {
                await this.uploadAudio(audioData);
                console.log(`Successfully uploaded: ${audioData.filename}`);
            } catch (error) {
                console.error(`Failed to upload ${audioData.filename}:`, error);
                // Re-add to queue for retry
                this.uploadQueue.unshift(audioData);
                break;
            }
        }
        
        this.isUploading = false;
    }

    /**
     * Upload audio to server
     */
    async uploadAudio(audioData) {
        const formData = new FormData();
        formData.append('audio', audioData.blob, audioData.filename);
        formData.append('timestamp', audioData.timestamp);

        // Log upload attempt
        console.log(`[AUDIO UPLOAD] Starting upload: ${audioData.filename}`);
        console.log(`[AUDIO UPLOAD] File size: ${(audioData.blob.size / 1024).toFixed(2)} KB`);
        console.log(`[AUDIO UPLOAD] Timestamp: ${audioData.timestamp}`);
        
        // Add log to index.html
        this.addUploadLog(`Uploading: ${audioData.filename} (${(audioData.blob.size / 1024).toFixed(2)} KB)`);

        try {
            const startTime = Date.now();
            const response = await fetch(this.uploadEndpoint, {
                method: 'POST',
                body: formData
            });

            const uploadTime = Date.now() - startTime;

            if (!response.ok) {
                const errorText = await response.text();
                console.error(`[AUDIO UPLOAD] Upload failed: ${response.statusText} - ${errorText}`);
                this.addUploadLog(`❌ Upload failed: ${audioData.filename} - ${response.statusText}`, 'error');
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const result = await response.json();
            console.log(`[AUDIO UPLOAD] Upload successful: ${audioData.filename}`);
            console.log(`[AUDIO UPLOAD] Server response:`, result);
            console.log(`[AUDIO UPLOAD] Upload time: ${uploadTime}ms`);
            
            // Add success log to index.html
            this.addUploadLog(`✅ Upload successful: ${audioData.filename} (${uploadTime}ms)`, 'success');

            return result;
        } catch (error) {
            console.error(`[AUDIO UPLOAD] Error uploading ${audioData.filename}:`, error);
            this.addUploadLog(`❌ Upload error: ${audioData.filename} - ${error.message}`, 'error');
            throw error;
        }
    }

    /**
     * Save recorded audio locally
     */
    async saveRecordedAudio() {
        if (this.recordedChunks.length === 0) {
            this.updateStatus('No recorded audio to save');
            return;
        }

        try {
            const blob = new Blob(this.recordedChunks, { type: 'audio/webm' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `recorded_audio_${new Date().toISOString().replace(/[:.]/g, '-')}.webm`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.updateStatus('Audio saved locally');
            this.addUploadLog('Audio saved locally', 'success');
        } catch (error) {
            console.error('Error saving audio:', error);
            this.updateStatus('Error saving audio: ' + error.message);
            this.addUploadLog(`❌ Save error: ${error.message}`, 'error');
        }
    }

    /**
     * Add upload log entry to the UI
     */
    addUploadLog(message, type = 'info') {
        // Create or find the upload log container
        let logContainer = document.getElementById('audio-upload-log-container');
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
        // Find the audio recorder section
        const audioRecorderSection = document.querySelector('#record-audio-btn').closest('.accordion-content');
        if (!audioRecorderSection) {
            console.error('Could not find audio recorder section for log container');
            return null;
        }

        // Create log container
        const logContainer = document.createElement('div');
        logContainer.id = 'audio-upload-log-container';
        logContainer.className = 'mt-4 p-3 bg-gray-900 rounded border border-gray-700 max-h-48 overflow-y-auto';
        
        // Add title
        const title = document.createElement('div');
        title.className = 'text-sm font-bold text-gray-300 mb-2';
        title.textContent = 'Audio Upload Log:';
        logContainer.appendChild(title);

        // Insert before the recording status
        const statusDiv = document.getElementById('audio-recording-status');
        if (statusDiv) {
            statusDiv.parentNode.insertBefore(logContainer, statusDiv.nextSibling);
        } else {
            audioRecorderSection.appendChild(logContainer);
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
        const recordBtn = document.getElementById('record-audio-btn');
        const saveBtn = document.getElementById('save-recorded-audio-btn');
        const statusDiv = document.getElementById('audio-recording-status');

        if (recordBtn) {
            if (this.isRecording) {
                recordBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <rect x="6" y="6" width="12" height="12" stroke-width="2"/>
                    </svg>
                    Stop Audio Recording
                `;
                recordBtn.classList.remove('bg-green-600', 'hover:bg-green-700');
                recordBtn.classList.add('bg-red-600', 'hover:bg-red-700');
            } else {
                recordBtn.innerHTML = `
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                    Start Audio Recording
                `;
                recordBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
                recordBtn.classList.add('bg-green-600', 'hover:bg-green-700');
            }
            
            recordBtn.disabled = false;
        }

        if (saveBtn) {
            saveBtn.disabled = this.recordedChunks.length === 0;
        }

        if (statusDiv) {
            const queueInfo = this.uploadQueue.length > 0 ? ` (Queue: ${this.uploadQueue.length})` : '';
            statusDiv.textContent = `Audio Recording: ${this.isRecording ? 'Active' : 'Not Active'}${queueInfo}`;
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
