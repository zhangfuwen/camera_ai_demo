# Audio Recording and Upload Feature

This document describes the audio recording and upload functionality that automatically records audio every 10 seconds and uploads it to the server.

## Features

- **Automatic Recording**: Records audio chunks from microphone every 10 seconds
- **Server Upload**: Automatically uploads recorded audio chunks to the `/upload_audio` endpoint
- **Timestamp Organization**: Audio files are saved with timestamps for chronological ordering
- **Queue Management**: Handles upload failures with retry mechanism
- **Manual Controls**: Start/stop recording and save audio locally
- **Real-time Logging**: Detailed upload status and progress tracking

## Implementation Details

### Frontend (JavaScript)

#### AudioRecorderController.js
- Location: `js/components/AudioRecorderController.js`
- Handles MediaRecorder API for audio capture from microphone
- Manages recording intervals and chunk processing
- Implements upload queue with retry logic
- Provides UI controls for manual recording control

#### Key Methods:
- `startRecording()`: Initiates audio recording from microphone
- `processCurrentRecording()`: Processes 10-second chunks and starts new recordings
- `uploadAudio()`: Sends audio data to server endpoint
- `processUploadQueue()`: Manages upload queue with retry mechanism

### Backend (Python)

#### food_detection_server.py
- Added `/upload_audio` endpoint for audio file uploads
- Creates `recorded_audio` directory for storage
- Saves audio files with timestamp prefixes for chronological ordering
- Handles file validation and secure filename generation

#### Endpoint Details:
- **URL**: `/upload_audio`
- **Method**: POST
- **Form Data**:
  - `audio`: Audio file (webm format)
  - `timestamp`: Timestamp string from client
- **Response**: JSON with success status, filename, and file details

## Usage

### Automatic Recording
1. Click "Start Audio Recording" in the Audio Recorder section
2. Grant microphone permission when prompted
3. The system will automatically:
   - Record 10-second audio chunks
   - Upload each chunk to the server
   - Continue recording until stopped

### Manual Controls
- **Start/Stop Audio Recording**: Toggle automatic recording
- **Save Audio Recording**: Download the most recent recording locally

### File Storage
- Audio files are saved in the `recorded_audio/` directory
- Filename format: `{timestamp}_{original_filename}.webm`
- Timestamps ensure chronological sorting

## Technical Specifications

### Audio Format
- **Codec**: Opus
- **Container**: WebM
- **Sample Rate**: 44.1 kHz
- **Chunk Duration**: 10 seconds
- **Audio Settings**: Echo cancellation, noise suppression, auto-gain control

### Upload Process
- **Method**: HTTP POST with FormData
- **Retry Logic**: Automatic retry on upload failures
- **Queue Management**: Prevents data loss during network issues

### Browser Compatibility
- Requires MediaRecorder API support
- Requires microphone access permission
- Tested with modern browsers (Chrome, Firefox, Safari)
- HTTPS required for microphone access in production

## Integration

### Main Application Integration
The AudioRecorderController is integrated into the main application:
- Imported in `js/main.js`
- Initialized with other controllers
- Event-driven architecture for UI interactions

### UI Controls
Located in the "Audio Recorder" section of the control panel:
- Record/Stop button with visual feedback
- Save button for local download
- Status display showing recording state and queue length
- Real-time upload log with detailed status messages

## Logging and Monitoring

### Frontend Logging
- Console logging for debugging
- UI log display showing upload progress
- Real-time status updates
- Error notifications and retry indicators

### Backend Logging
- Detailed request/response logging
- File processing information
- Error tracking with stack traces
- Directory management logs

### Log Examples:
```
[AUDIO UPLOAD] Request received at 2025-01-25T12:47:00.123456
[AUDIO UPLOAD] Audio file received: audio_2025-01-25T12-47-00.webm
[AUDIO UPLOAD] File size: 1024 bytes (1.00 KB)
[AUDIO UPLOAD] Upload completed successfully
```

## Troubleshooting

### Common Issues
1. **Microphone Permission**: Ensure microphone permissions are granted
2. **HTTPS Requirement**: Some browsers require HTTPS for microphone access
3. **Upload Failures**: Check server connectivity and endpoint availability
4. **File Storage**: Ensure write permissions for `recorded_audio/` directory

### Debug Information
- Browser console: JavaScript errors and upload status
- Server logs: Upload endpoint activity and file storage details
- Network tab: HTTP requests and responses for debugging

### Error Handling
- Automatic retry on upload failures
- Graceful handling of microphone permission denial
- Queue management for interrupted uploads
- User-friendly error messages

## Security Considerations

### File Upload Security
- Filename sanitization using `werkzeug.utils.secure_filename`
- File type validation
- Size limitations
- Directory traversal protection

### Privacy Considerations
- Microphone access requires explicit user permission
- Audio data is processed locally before upload
- No audio recording without user initiation
- Clear visual indicators when recording is active

## Future Enhancements

Potential improvements for the audio recording feature:
- Configurable recording duration
- Audio quality settings
- Multiple audio format support
- Real-time audio visualization
- Audio level monitoring
- Background recording capabilities
- Audio compression options
- Recording metadata storage

## Comparison with Video Recording

| Feature | Audio Recording | Video Recording |
|---------|----------------|------------------|
| Source | Microphone | Main camera stream |
| Format | WebM (Opus) | WebM (VP8+Opus) |
| Duration | 10 seconds | 10 seconds |
| Storage | `recorded_audio/` | `recorded_videos/` |
| Endpoint | `/upload_audio` | `/upload_video` |
| Permissions | Microphone access | Camera access (shared) |

## API Reference

### AudioRecorderController Methods

#### `initialize()`
Initializes the audio recorder controller and sets up event listeners.

#### `startRecording()`
Starts recording audio from the microphone with automatic 10-second chunks.

#### `stopRecording()`
Stops recording and cleans up audio resources.

#### `uploadAudio(audioData)`
Uploads audio data to the server with detailed logging and error handling.

#### `saveRecordedAudio()`
Saves the most recent audio recording locally for download.

#### `addUploadLog(message, type)`
Adds log entries to the UI with different types (info, success, error).

### Server Endpoints

#### POST `/upload_audio`
Uploads audio files to the server with timestamp organization.

**Request:**
```
Content-Type: multipart/form-data
audio: [audio file]
timestamp: [ISO timestamp]
```

**Response:**
```json
{
  "success": true,
  "filename": "2025-01-25T12-47-00_audio.webm",
  "timestamp": "2025-01-25T12:47:00",
  "path": "recorded_audio/2025-01-25T12-47-00_audio.webm",
  "size": 1024
}
```
