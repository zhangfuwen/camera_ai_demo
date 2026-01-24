# Real-time Streaming Channel Implementation

This document describes the real-time streaming channel between server and client that pushes updates for 5 types of information to the UI.

## Overview

The streaming system uses SocketIO to establish a real-time bidirectional communication channel between the server and client. The server automatically pushes updates to 5 different status overlays in the UI based on various triggers and timers.

## Architecture

### Server-Side Components

#### 1. SocketIO Integration
- **Library**: Flask-SocketIO
- **Endpoint**: WebSocket connection on `/socket.io/`
- **Events**: Real-time message broadcasting to connected clients

#### 2. Data Sources & Triggers

| Data Type | Trigger | Target UI Element | Update Frequency |
|-----------|---------|------------------|------------------|
| Sensor Values | Timer (1 second) | `#status_overlay` | Every 1 second |
| User Emotions | Timer (3 seconds) | `#status_overlay2` | Every 3 seconds |
| Audio Detection | `/upload_audio` + Gemini | `#status_overlay3` | On audio upload |
| Video Detection | `/upload_video` + Gemini | `#status_overlay4` | On video upload |
| Overall Status | Timer (20 seconds) | `#status_overlay5` | Every 20 seconds |

#### 3. Gemini Model Integration

**Video Analysis (`/upload_video`)**:
- Analyzes uploaded video content
- Updates `#status_overlay4` with activity detection
- Updates `#status_overlay2` with emotion/expression analysis
- Simulated API delay: 2 seconds

**Audio Analysis (`/upload_audio`)**:
- Analyzes uploaded audio content  
- Updates `#status_overlay3` with audio activity detection
- Simulated API delay: 1.5 seconds

### Client-Side Components

#### StreamController.js
- **Connection**: SocketIO client connection
- **Message Handling**: Routes updates to appropriate UI elements
- **Reconnection**: Automatic reconnection with exponential backoff
- **Heartbeat**: Keeps connection alive with 30-second intervals

## Implementation Details

### Server Implementation

#### WebSocket Events
```python
@socketio.on('connect')
def handle_connect():
    # Client connected
    
@socketio.on('disconnect') 
def handle_disconnect():
    # Client disconnected
    
@socketio.on('heartbeat')
def handle_heartbeat():
    # Connection health check
```

#### Message Broadcasting
```python
def broadcast_to_clients(message_type, content):
    message = {
        'type': message_type,
        'content': content,
        'timestamp': datetime.now().isoformat()
    }
    socketio.emit('stream_update', message)
```

#### Timer Functions
- **sensor_update_timer()**: Generates fake sport watch data every 1 second
- **emotion_update_timer()**: Generates emotion data every 3 seconds  
- **overall_status_timer()**: Generates user status summary every 20 seconds

### Client Implementation

#### Connection Management
```javascript
this.socket = io();
this.socket.on('connect', () => { /* connected */ });
this.socket.on('stream_update', (data) => { /* handle update */ });
```

#### UI Updates
```javascript
switch (data.type) {
    case 'sensor_update':
        this.updateStatusOverlay('#status_overlay', data.content);
        break;
    // ... other cases
}
```

## Data Flow

### 1. Timer-Based Updates
```
Timer Trigger → Generate Data → Broadcast to Clients → Update UI
```

### 2. Upload-Based Updates
```
File Upload → Save File → Gemini Analysis → Broadcast Results → Update UI
```

### 3. Connection Management
```
Client Connect → Register Client → Start Updates → Monitor Connection
```

## UI Elements

### Status Overlays

1. **#status_overlay** (Sensor Values)
   - Energy level
   - Sleep hours
   - Sport/steps count
   - Blood pressure
   - Heart rate
   - Calories

2. **#status_overlay2** (User Emotions)
   - Detected emotion
   - Confidence level
   - Detection timestamp

3. **#status_overlay3** (Audio Detection)
   - Activity type
   - Volume level
   - Duration

4. **#status_overlay4** (Video Detection)
   - Activity description
   - Detected objects
   - Movement level

5. **#status_overlay5** (Overall Status)
   - User status summary
   - Last update timestamp

### Connection Status Indicator
- **Element**: `#stream-connection-status`
- **States**: Connected (green), Disconnected (red), Error (orange)
- **Location**: Bottom of #status_overlay

## Message Format

### Server → Client Messages
```json
{
    "type": "sensor_update|emotion_update|audio_detect|video_detect|overall_status",
    "content": "<html_content>",
    "timestamp": "2025-01-25T01:07:00.000000"
}
```

### HTML Content Examples

#### Sensor Values
```html
<div class="text-green-500">
    <div>Energy: 75</div>
    <div>Sleep: 7.23 Hours</div>
    <div>Sport: 5432 Steps</div>
    <div>Blood Pressure: 85</div>
    <div>HeartRate: 72</div>
    <div>Calories: 425</div>
</div>
```

#### Emotion Detection
```html
<div class="text-green-500">
    <div>Emotion: Focused</div>
    <div>Confidence: 0.87</div>
    <div>Detected: 01:07:05</div>
</div>
```

## Configuration

### Server Configuration
- **Port**: 5000
- **Host**: 0.0.0.0 (all interfaces)
- **CORS**: Enabled for all origins
- **Debug**: Disabled for production

### Client Configuration
- **Reconnect Attempts**: 5 maximum
- **Reconnect Interval**: 3000ms
- **Heartbeat Interval**: 30 seconds

## Timer Configuration

| Timer | Duration | Purpose | Data Source |
|-------|----------|---------|-------------|
| Sensor Update | 1 second | Sport watch data | Random generation |
| Emotion Update | 3 seconds | User emotions | Random generation |
| Overall Status | 20 seconds | User summary | Random generation |

## Gemini Integration (Placeholder)

Currently implemented with mock data. Real integration would require:

### Video Analysis
```python
# Placeholder for real Gemini API call
async def analyze_video_with_gemini(video_path):
    # TODO: Implement actual Gemini API integration
    # - Upload video to Gemini
    # - Get activity analysis
    # - Get emotion/expression analysis
    # - Return structured results
```

### Audio Analysis  
```python
# Placeholder for real Gemini API call
async def analyze_audio_with_gemini(audio_path):
    # TODO: Implement actual Gemini API integration
    # - Upload audio to Gemini
    # - Get activity detection
    # - Get speech analysis
    # - Return structured results
```

## Error Handling

### Connection Errors
- Automatic reconnection with exponential backoff
- Connection status indicator in UI
- Graceful degradation when disconnected

### Processing Errors
- Timer error handling with retry logic
- Gemini analysis error handling
- Client message parsing error handling

## Performance Considerations

### Server-Side
- Background threads for timers
- Non-blocking Gemini analysis
- Efficient client tracking

### Client-Side
- Efficient DOM updates
- Connection state management
- Memory leak prevention

## Security Considerations

### Current Implementation
- CORS enabled for all origins (development)
- No authentication required
- File upload validation

### Production Recommendations
- Implement authentication/authorization
- Restrict CORS to specific origins
- Add rate limiting
- Validate all user inputs
- Secure Gemini API keys

## Testing

### Manual Testing
1. Start server: `python3 food_detection_server.py`
2. Open web application: `http://localhost:5000`
3. Verify connection status turns green
4. Observe automatic updates in all 5 overlays
5. Test video/audio upload triggers

### Expected Behavior
- Connection status: "Stream: Connected" (green)
- Sensor values: Update every 1 second
- Emotions: Update every 3 seconds  
- Audio detection: Updates on audio upload
- Video detection: Updates on video upload
- Overall status: Updates every 20 seconds

## Troubleshooting

### Common Issues

1. **SocketIO Connection Failed**
   - Check flask-socketio installation
   - Verify port 5000 is available
   - Check browser console for errors

2. **No Updates in UI**
   - Verify server timers are running
   - Check browser console for message reception
   - Verify UI element selectors

3. **Upload Triggers Not Working**
   - Check file upload functionality
   - Verify Gemini analysis threads
   - Check server logs for errors

### Debug Information

#### Server Logs
```
[STREAM] Client connected: <session_id>
[STREAM] Broadcasted sensor_update to 1 clients
[GEMINI] Started video analysis for: <filename>
```

#### Browser Console
```
SocketIO connected successfully
Received message: {type: "sensor_update", content: "..."}
Updated #status_overlay with new content
```

## Future Enhancements

### Gemini Integration
- Real Google Gemini API integration
- Configurable analysis parameters
- Batch processing support
- Error handling and retry logic

### Advanced Features
- User authentication
- Multiple user support
- Historical data tracking
- Custom alert configurations
- Mobile app support

### Performance Optimizations
- WebSocket connection pooling
- Efficient data serialization
- Caching mechanisms
- Load balancing support

## Files Modified

### Server Files
- `food_detection_server.py` - Added SocketIO and streaming logic
- `requirements.txt` - Added flask-socketio dependency

### Client Files  
- `js/components/StreamController.js` - New SocketIO client controller
- `js/main.js` - Integrated StreamController
- `index.html` - Added SocketIO library and connection status

### Documentation
- `STREAMING_README.md` - This comprehensive documentation

## Dependencies

### New Dependencies
- `flask-socketio` - WebSocket support for Flask
- `python-socketio` - SocketIO Python implementation
- `socket.io-client` - Client-side SocketIO library (CDN)

### Existing Dependencies
- All existing dependencies remain unchanged

## API Reference

### Server Events

#### `connect`
Client connects to server.

#### `disconnect` 
Client disconnects from server.

#### `heartbeat`
Connection health check.

#### `stream_update`
Broadcast message to all connected clients.

### Client Events

#### `connect`
Automatic connection to server.

#### `stream_update`
Handle incoming update messages.

#### `disconnect`
Handle disconnection events.

This streaming implementation provides a robust foundation for real-time data updates and can be extended with additional features as needed.
