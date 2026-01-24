/**
 * Stream Controller
 * Handles SocketIO connection for real-time updates from server
 */
export class StreamController {
    constructor() {
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectInterval = 3000;
        this.isConnected = false;
        this.heartbeatInterval = null;
    }

    /**
     * Initialize the stream controller
     */
    initialize() {
        console.log('Initializing Stream Controller...');
        this.connect();
    }

    /**
     * Connect to SocketIO server
     */
    connect() {
        try {
            console.log('Connecting to SocketIO server...');
            this.socket = io();

            this.socket.on('connect', () => {
                console.log('SocketIO connected successfully');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.startHeartbeat();
                this.updateConnectionStatus('Connected');
            });

            this.socket.on('disconnect', (reason) => {
                console.log('SocketIO disconnected:', reason);
                this.isConnected = false;
                this.stopHeartbeat();
                this.updateConnectionStatus('Disconnected');
                this.handleReconnect();
            });

            this.socket.on('connect_error', (error) => {
                console.error('SocketIO connection error:', error);
                this.updateConnectionStatus('Error');
                this.handleReconnect();
            });

            this.socket.on('stream_update', (data) => {
                this.handleMessage(data);
            });

            this.socket.on('status', (data) => {
                console.log('Status message:', data);
            });

            this.socket.on('heartbeat_response', (data) => {
                // Handle heartbeat response
                console.log('Heartbeat response:', data);
            });

        } catch (error) {
            console.error('Failed to create SocketIO connection:', error);
            this.handleReconnect();
        }
    }

    /**
     * Handle incoming SocketIO messages
     */
    handleMessage(data) {
        try {
            console.log('Received message:', data);

            switch (data.type) {
                case 'sensor_update':
                    this.updateStatusOverlay('#status-overlay', data.content);
                    break;
                case 'emotion_update':
                    this.updateStatusOverlay('#status-overlay2', data.content);
                    break;
                case 'audio_detect':
                    this.updateStatusOverlay('#status-overlay3', data.content);
                    break;
                case 'video_detect':
                    this.updateStatusOverlay('#status-overlay4', data.content);
                    break;
                case 'overall_status':
                    this.updateStatusOverlay('#status-overlay5', data.content);
                    break;
                case 'heartbeat':
                    // Handle heartbeat response
                    break;
                default:
                    console.warn('Unknown message type:', data.type);
            }
        } catch (error) {
            console.error('Error parsing SocketIO message:', error);
        }
    }

    /**
     * Update status overlay with new content
     */
    updateStatusOverlay(selector, content) {
        const element = document.querySelector(selector);
        if (element) {
            element.innerHTML = content;
            console.log(`Updated ${selector} with new content`);
        } else {
            console.warn(`Element not found: ${selector}`);
        }
    }

    /**
     * Handle SocketIO reconnection
     */
    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            
            setTimeout(() => {
                this.connect();
            }, this.reconnectInterval);
        } else {
            console.error('Max reconnection attempts reached');
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
