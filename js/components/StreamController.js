/**
 * Stream Controller
 * Handles SocketIO connection for real-time updates from server
 */
import { Logger } from '../utils/logger.js';

export class StreamController {
    constructor() {
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 3000;
        this.isConnected = false;
        this.heartbeatInterval = null;
        this.logger = new Logger('StreamController', 'INFO');
    }

    /**
     * Initialize the stream controller
     */
    initialize() {
        this.logger.info('Initializing Stream Controller...');
        this.connect();
    }

    /**
     * Connect to SocketIO server
     */
    connect() {
        try {
            this.logger.info('Connecting to SocketIO server...');
            this.socket = io();

            this.socket.on('connect', () => {
                this.logger.info('SocketIO connected successfully');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.startHeartbeat();
                this.updateConnectionStatus('Connected');
            });

            this.socket.on('disconnect', (reason) => {
                this.logger.warning(`SocketIO disconnected: ${reason}`);
                this.isConnected = false;
                this.stopHeartbeat();
                this.updateConnectionStatus('Disconnected');
                this.handleReconnect();
            });

            this.socket.on('connect_error', (error) => {
                this.logger.error(`SocketIO connection error: ${error.message}`);
                this.updateConnectionStatus('Error');
                this.handleReconnect();
            });

            this.socket.on('stream_update', (data) => {
                this.handleMessage(data);
            });

            this.socket.on('status', (data) => {
                this.logger.debug('Status message received:', data);
            });

            this.socket.on('heartbeat_response', (data) => {
                // Handle heartbeat response
                this.logger.verbose('Heartbeat response:', data);
            });

        } catch (error) {
            this.logger.error(`Failed to create SocketIO connection: ${error.message}`);
            this.handleReconnect();
        }
    }

    /**
     * Handle incoming SocketIO messages
     */
    handleMessage(data) {
        try {
            this.logger.debug(`Received ${data.type} message`);

            switch (data.type) {
                case 'sensor_update':
                    this.updateStatusOverlay('#bio-info-container', data.content);
                    break;
                case 'emotion_update':
                    this.updateStatusOverlay('#emotion-info-container', data.content);
                    break;
                case 'audio_detect':
                    this.updateStatusOverlay('#voice-info-container', data.content);
                    break;
                case 'video_detect':
                    this.updateStatusOverlay('#food-info-container', data.content);
                    break;
                case 'overall_status':
                    this.updateStatusOverlay('#overall-info-container', data.content);
                    break;
                case 'heartbeat':
                    // Handle heartbeat response
                    break;
                default:
                    this.logger.warning(`Unknown message type: ${data.type}`);
            }
        } catch (error) {
            this.logger.error(`Error parsing SocketIO message: ${error.message}`);
        }
    }

    /**
     * Update status overlay with new content and flip animation
     */
    updateStatusOverlay(selector, content) {
        const element = document.querySelector(selector);
        if (element) {
            // Check if flip animation is enabled
            const flipAnimationEnabled = document.getElementById('flip-animation-toggle')?.checked ?? true;
            const animationStyle = document.getElementById('animation-style-select')?.value ?? 'card-flip';
            
            // if (flipAnimationEnabled) {
            //     // Add animation class based on selected style
            //     element.classList.add(animationStyle);
            //
            //     // Update content after a short delay to sync with animation
            //     const delay = animationStyle === 'pulse' ? 0 : 300;
            //     setTimeout(() => {
            //         // Use textContent for plain text and style for better formatting
            //         element.innerHTML = content;
            //         // element.textContent = content;
            //         // element.style.whiteSpace = 'pre-line';
            //         // element.style.fontFamily = 'monospace';
            //         // element.style.fontSize = '11px';
            //         // element.style.lineHeight = '1.3';
            //         // element.style.color = '#22c55e';
            //
            //         this.logger.debug(`Updated ${selector} with plain text content (${animationStyle} animation)`);
            //     }, delay);
            //
            //     // Remove animation class after animation completes
            //     const animationDuration = animationStyle === 'pulse' ? 400 : 600;
            //     setTimeout(() => {
            //         element.classList.remove(animationStyle);
            //     }, animationDuration);
            // } else {
                // Update content immediately without animation
                element.innerHTML = content;
                // element.textContent = content;
                // element.style.whiteSpace = 'pre-line';
                // element.style.fontFamily = 'monospace';
                // element.style.fontSize = '11px';
                // element.style.lineHeight = '1.3';
                // element.style.color = '#22c55e';
                
            //     this.logger.debug(`Updated ${selector} with plain text content (no animation)`);
            // }
            
        } else {
            this.logger.warning(`Element not found: ${selector}`);
        }
    }
    
    /**
     * Update status overlay with pulse animation (alternative effect)
     */
    updateStatusOverlayPulse(selector, content) {
        const element = document.querySelector(selector);
        if (element) {
            // Check if flip animation is enabled
            const flipAnimationEnabled = document.getElementById('flip-animation-toggle')?.checked ?? true;
            
            if (flipAnimationEnabled) {
                // Add pulse animation class
                element.classList.add('status-pulse');
                
                // Update content immediately
                element.textContent = content;
                element.style.whiteSpace = 'pre-line';
                element.style.fontFamily = 'monospace';
                element.style.fontSize = '11px';
                element.style.lineHeight = '1.3';
                element.style.color = '#22c55e';
                
                this.logger.debug(`Updated ${selector} with plain text content (pulse effect)`);
                
                // Remove animation class after animation completes
                setTimeout(() => {
                    element.classList.remove('status-pulse');
                }, 400);
            } else {
                // Update content immediately without animation
                element.textContent = content;
                element.style.whiteSpace = 'pre-line';
                element.style.fontFamily = 'monospace';
                element.style.fontSize = '11px';
                element.style.lineHeight = '1.3';
                element.style.color = '#22c55e';
                
                this.logger.debug(`Updated ${selector} with plain text content (no animation)`);
            }
            
        } else {
            this.logger.warning(`Element not found: ${selector}`);
        }
    }
    
    /**
     * Update status overlay with flip animation (Y-axis rotation)
     */
    updateStatusOverlayFlip(selector, content) {
        const element = document.querySelector(selector);
        if (element) {
            // Check if flip animation is enabled
            const flipAnimationEnabled = document.getElementById('flip-animation-toggle')?.checked ?? true;
            
            if (flipAnimationEnabled) {
                // Add flip animation class
                element.classList.add('flipping');
                
                // Update content after a short delay to sync with animation
                setTimeout(() => {
                    element.textContent = content;
                    element.style.whiteSpace = 'pre-line';
                    element.style.fontFamily = 'monospace';
                    element.style.fontSize = '11px';
                    element.style.lineHeight = '1.3';
                    element.style.color = '#22c55e';
                    
                    this.logger.debug(`Updated ${selector} with plain text content (flip effect)`);
                }, 300); // Halfway through the animation
                
                // Remove animation class after animation completes
                setTimeout(() => {
                    element.classList.remove('flipping');
                }, 600);
            } else {
                // Update content immediately without animation
                element.textContent = content;
                element.style.whiteSpace = 'pre-line';
                element.style.fontFamily = 'monospace';
                element.style.fontSize = '11px';
                element.style.lineHeight = '1.3';
                element.style.color = '#22c55e';
                
                this.logger.debug(`Updated ${selector} with plain text content (no animation)`);
            }
            
        } else {
            this.logger.warning(`Element not found: ${selector}`);
        }
    }

    /**
     * Handle SocketIO reconnection
     */
    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.logger.info(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            
            setTimeout(() => {
                this.connect();
            }, this.reconnectInterval);
        } else {
            this.logger.error('Max reconnection attempts reached');
            this.updateConnectionStatus('Connection Failed');
        }
    }

    /**
     * Start heartbeat to keep connection alive
     */
    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected && this.socket) {
                this.socket.emit('heartbeat');
            }
        }, 30000); // Send heartbeat every 30 seconds
    }

    /**
     * Stop heartbeat
     */
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    /**
     * Update connection status display
     */
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('stream-connection-status');
        if (statusElement) {
            statusElement.textContent = `Stream: ${status}`;
            statusElement.className = `text-xs px-2 py-1 rounded ${
                status === 'Connected' ? 'bg-green-600' :
                status === 'Disconnected' ? 'bg-red-600' :
                status === 'Error' ? 'bg-orange-600' :
                'bg-gray-600'
            } text-white mt-2`;
        }
    }

    /**
     * Send message to server
     */
    sendMessage(message) {
        if (this.isConnected && this.socket) {
            this.socket.emit('client_message', message);
        } else {
            console.warn('SocketIO not connected, message not sent:', message);
        }
    }

    /**
     * Disconnect SocketIO
     */
    disconnect() {
        this.stopHeartbeat();
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        this.isConnected = false;
    }

    /**
     * Get connection status
     */
    getConnectionStatus() {
        return {
            isConnected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts
        };
    }
}
