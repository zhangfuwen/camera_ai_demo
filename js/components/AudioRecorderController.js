/**
 * Audio Recorder Controller
 * Handles recording audio every 10 seconds and uploading to server
 */
import { Logger } from '../utils/logger.js';

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
        this.audioDevices = [];
        this.selectedAudioDeviceId = null;
        this.logger = new Logger('AudioRecorderController', 'INFO');
    }

    /**
     * Initialize the audio recorder controller
     */
    initialize() {
        this.logger.info('Initializing Audio Recorder Controller...');
        this.setupEventListeners();
        this.updateUI();
        this.getAudioDevices();
        // Proactively request microphone permission so device labels and recording work properly
        this.initializeMicrophonePermission();
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

        // Allow manual microphone permission request via custom event
        document.addEventListener('requestMicrophonePermission', () => {
            this.initializeMicrophonePermission();
        });

        // Audio source selection
        const audioSourceSelect = document.getElementById('audio-source-select');
        if (audioSourceSelect) {
            audioSourceSelect.addEventListener('change', (e) => {
                this.selectedAudioDeviceId = e.target.value;
                this.logger.debug(`Selected audio device: ${this.selectedAudioDeviceId}`);
            });
        }

        // Refresh audio sources button
        const refreshBtn = document.getElementById('refresh-audio-sources-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.getAudioDevices();
            });
        }
    }

    /**
     * Request microphone permission
     */
    async initializeMicrophonePermission() {
        try {
            this.logger.debug('Requesting microphone permission...');
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            // Stop the stream immediately after getting permission
            stream.getTracks().forEach(track => track.stop());

            this.logger.info('Microphone permission granted');
            this.updateStatus('Microphone permission granted');
            // Refresh audio device list to populate labels after permission
            await this.getAudioDevices();
            return true;
        } catch (error) {
            this.logger.error(`Error requesting microphone permission: ${error.message}`);
            this.updateStatus('Error: Microphone permission denied');
            this.addUploadLog('Error requesting microphone permission: ' + error.message, 'error');
            return false;
        }
    }

    /**
     * Get available audio devices
     */
    async getAudioDevices() {
        try {
            this.logger.debug('Getting audio devices...');
            const devices = await navigator.mediaDevices.enumerateDevices();
            
            // Filter audio input devices
            this.audioDevices = devices.filter(device => device.kind === 'audioinput');
            
            this.logger.debug(`Found ${this.audioDevices.length} audio devices`);
            
            // Update UI with device list
            this.updateAudioDeviceList();
            
        } catch (error) {
            this.logger.error(`Error getting audio devices: ${error.message}`);
            this.addUploadLog('Error getting audio devices: ' + error.message, 'error');
        }
    }

    /**
     * Update audio device list in UI
     */
    updateAudioDeviceList() {
        const audioSourceSelect = document.getElementById('audio-source-select');
        const refreshBtn = document.getElementById('refresh-audio-sources-btn');
        
        if (audioSourceSelect) {
            // Clear existing options
            audioSourceSelect.innerHTML = '<option value="">-- Select an audio source --</option>';
            
            // Add device options
            this.audioDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.textContent = device.label || `Microphone ${index + 1}`;
                audioSourceSelect.appendChild(option);
            });
            
            // Enable the select if devices are available
            audioSourceSelect.disabled = this.audioDevices.length === 0;
            
            // Auto-select first device if none selected
            if (this.audioDevices.length > 0 && !this.selectedAudioDeviceId) {
                this.selectedAudioDeviceId = this.audioDevices[0].deviceId;
                audioSourceSelect.value = this.selectedAudioDeviceId;
            }
        }
        
        // Enable refresh button
        if (refreshBtn) {
            refreshBtn.disabled = false;
        }
        
        // Enable record button if we have devices
        const recordBtn = document.getElementById('record-audio-btn');
        if (recordBtn) {
            recordBtn.disabled = this.audioDevices.length === 0;
        }
        
        this.addUploadLog(`Found ${this.audioDevices.length} audio device(s)`, 'info');
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
            if (!this.selectedAudioDeviceId) {
                throw new Error('No audio device selected');
            }
            
            // Request microphone access with specific device
            this.logger.debug(`Requesting microphone access for device: ${this.selectedAudioDeviceId}`);
            
            const constraints = {
                audio: {
                    deviceId: this.selectedAudioDeviceId ? { exact: this.selectedAudioDeviceId } : undefined,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 44100
                }
            };
            
            this.audioStream = await navigator.mediaDevices.getUserMedia(constraints);
            
            this.logger.info('Microphone access granted');
            
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

            this.logger.info('Audio recording started');
            this.updateUI();
            this.updateStatus('Audio recording: Active');
            this.addUploadLog('Audio recording started', 'success');
            
            // Set up interval to process recordings every 10 seconds
            this.recordingInterval = setInterval(() => {
                this.processCurrentRecording();
            }, this.recordingDuration);

        } catch (error) {
            this.logger.error(`Error starting audio recording: ${error.message}`);
            this.updateStatus('Error starting audio recording: ' + error.message);
            this.addUploadLog(`❌ Error starting recording: ${error.message}`, 'error');
        }
    }

    /**
     * Stop audio recording
     */
    stopRecording() {
        this.logger.info('Stopping audio recording...');
        
        // Clear the recording interval first
        if (this.recordingInterval) {
            clearInterval(this.recordingInterval);
            this.recordingInterval = null;
        }

        // Stop the MediaRecorder
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.logger.debug('Stopping Audio MediaRecorder...');
            this.mediaRecorder.stop();
        }

        // Stop the audio stream
        if (this.audioStream) {
            this.logger.debug('Stopping audio stream...');
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
        }

        this.isRecording = false;
        this.logger.info('Audio recording stopped');
        
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
            
            // Update voice information to show processing
            const timestamp = new Date().toLocaleTimeString();

            // Stop current MediaRecorder
            this.mediaRecorder.stop();
            
            // Start a new recording immediately
            setTimeout(() => {
                if (this.isRecording) {
                    this.startNewRecordingChunk();
                    // Update voice info to show new recording started
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
        // Find the upload log container that should already exist in the HTML
        let logContainer = document.getElementById('audio-upload-log-container');
        if (!logContainer) {
            console.error('Could not find audio upload log container');
            return null;
        }
        
        // Make it visible
        // logContainer.classList.remove('hidden');
        
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
                    Stop Audio Recording
                `;
                recordBtn.classList.remove('bg-green-600', 'hover:bg-green-700');
                recordBtn.classList.add('bg-red-600', 'hover:bg-red-700');
            } else {
                recordBtn.innerHTML = `
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
