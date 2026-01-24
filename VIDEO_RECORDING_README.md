# Video Recording and Upload Feature

This document describes the video recording and upload functionality that automatically records video from the #main-video element every 10 seconds and uploads it to the server.

## Features

- **Automatic Recording**: Records video chunks from the main camera feed every 10 seconds
- **Server Upload**: Automatically uploads recorded video chunks to the `/upload_video` endpoint
- **Timestamp Organization**: Videos are saved with timestamps for chronological ordering
- **Queue Management**: Handles upload failures with retry mechanism
- **Manual Controls**: Start/stop recording and save video locally

## Implementation Details

### Frontend (JavaScript)

#### VideoRecorderController.js
- Location: `js/components/VideoRecorderController.js`
- Handles MediaRecorder API for video capture
- Manages recording intervals and chunk processing
- Implements upload queue with retry logic
- Provides UI controls for manual recording control

#### Key Methods:
- `startRecording()`: Initiates video recording from #main-video stream
- `processCurrentRecording()`: Processes 10-second chunks and starts new recordings
- `uploadVideo()`: Sends video data to server endpoint
- `processUploadQueue()`: Manages upload queue with retry mechanism

### Backend (Python)

#### food_detection_server.py
- Added `/upload_video` endpoint for video file uploads
- Creates `recorded_videos` directory for storage
- Saves videos with timestamp prefixes for chronological ordering
- Handles file validation and secure filename generation

#### Endpoint Details:
- **URL**: `/upload_video`
- **Method**: POST
- **Form Data**:
  - `video`: Video file (webm format)
  - `timestamp`: Timestamp string from client
- **Response**: JSON with success status, filename, and file details

## Usage

### Automatic Recording
1. Start the main camera feed
2. Click "Start Recording" in the Video Recorder section
3. The system will automatically:
   - Record 10-second video chunks
   - Upload each chunk to the server
   - Continue recording until stopped

### Manual Controls
- **Start/Stop Recording**: Toggle automatic recording
- **Save Recording**: Download the most recent recording locally

### File Storage
- Videos are saved in the `recorded_videos/` directory
- Filename format: `{timestamp}_{original_filename}.webm`
- Timestamps ensure chronological sorting

## Technical Specifications

### Video Format
- **Codec**: VP8 (video), Opus (audio)
- **Container**: WebM
- **Chunk Duration**: 10 seconds
- **Quality**: Default browser settings

### Upload Process
- **Method**: HTTP POST with FormData
- **Retry Logic**: Automatic retry on upload failures
- **Queue Management**: Prevents data loss during network issues

### Browser Compatibility
- Requires MediaRecorder API support
- Tested with modern browsers (Chrome, Firefox, Safari)
- HTTPS required for camera access in production

## Integration

### Main Application Integration
The VideoRecorderController is integrated into the main application:
- Imported in `js/main.js`
- Initialized with other controllers
- Event-driven architecture for UI interactions

### UI Controls
Located in the "Video Recorder" section of the control panel:
- Record/Stop button with visual feedback
- Save button for local download
- Status display showing recording state and queue length

## Troubleshooting

### Common Issues
1. **Camera Permission**: Ensure camera permissions are granted
2. **HTTPS Requirement**: Some browsers require HTTPS for camera access
3. **Upload Failures**: Check server connectivity and endpoint availability
4. **File Storage**: Ensure write permissions for `recorded_videos/` directory

### Debug Information
- Browser console: JavaScript errors and upload status
- Server logs: Upload endpoint activity and file storage details
- Network tab: HTTP requests and responses for debugging

## Future Enhancements

Potential improvements for the video recording feature:
- Configurable recording duration
- Video quality settings
- Compression options
- Batch upload management
- Recording metadata storage
- Video preview functionality
