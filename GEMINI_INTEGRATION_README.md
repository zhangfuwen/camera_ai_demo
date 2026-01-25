# Gemini AI Integration Implementation

This document describes the real Gemini AI integration for video and audio analysis in the camera AI demo application.

## Overview

The application now uses Google's Gemini 2.5 Flash model to analyze uploaded video and audio files, providing intelligent insights about user activities, emotions, and environment.

## Implementation Details

### Gemini Client Configuration

```python
from google import genai

# Initialize Gemini client
gemini_client = genai.Client(
    api_key=os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY'),
    http_options={"base_url": "https://api.apiyi.com"}
)
```

### Video Analysis

#### Trigger
- **Endpoint**: `/upload_video`
- **File Types**: WebM video files
- **Target UI**: `#status_overlay4` (activity) + `#status_overlay2` (emotions)

#### Analysis Process
1. **Upload**: Video file uploaded to Gemini API
2. **Analysis**: Content analysis with structured prompts
3. **Parsing**: Response parsed into structured data
4. **Broadcast**: Results sent to connected clients
5. **Cleanup**: Uploaded file deleted from Gemini servers

#### Analysis Prompt
```python
"""请分析这个视频，提供以下信息：
1. 视频中主要的活动内容
2. 检测到的对象或物品
3. 用户的动作或行为
4. 整体的活动强度（低/中/高）
5. 用户的情绪状态或表情

请用简洁的中文回答，每个项目用一行表示。"""
```

#### Output Format
```html
<div class="text-green-500">
    <div><strong>活动:</strong> 用户正在电脑前工作</div>
    <div><strong>对象:</strong> 电脑, 键盘, 鼠标</div>
    <div><strong>活动强度:</strong> 中</div>
    <div class="text-xs mt-1">分析时间: 14:30:25</div>
</div>
```

### Audio Analysis

#### Trigger
- **Endpoint**: `/upload_audio`
- **File Types**: WebM audio files
- **Target UI**: `#status_overlay3` (audio detection)

#### Analysis Process
1. **Upload**: Audio file uploaded to Gemini API
2. **Transcription**: Audio content transcribed and analyzed
3. **Analysis**: Content analysis with structured prompts
4. **Broadcast**: Results sent to connected clients
5. **Cleanup**: Uploaded file deleted from Gemini servers

#### Analysis Prompt
```python
"""请分析这段音频，提供以下信息：
1. 音频中的主要活动（说话、静音、背景噪音等）
2. 音量水平（低/中/高）
3. 持续时间估算
4. 如果有语音，简要总结主要内容
5. 整体的音频环境描述

请用简洁的中文回答，每个项目用一行表示。"""
```

#### Output Format
```html
<div class="text-green-500">
    <div><strong>活动:</strong> 说话</div>
    <div><strong>音量:</strong> 中</div>
    <div><strong>持续时间:</strong> 约10秒</div>
    <div class="text-xs mt-1">分析时间: 14:30:25</div>
</div>
```

## Configuration

### Environment Variables
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### API Provider
- **Base URL**: `https://api.apiyi.com`
- **Model**: `gemini-2.5-flash`
- **Authentication**: API key

## Error Handling

### Fallback Mechanism
If Gemini client initialization fails or API calls encounter errors:
- **Video Analysis**: Falls back to mock data generation
- **Audio Analysis**: Falls back to mock data generation
- **Logging**: Errors logged with `[GEMINI]` prefix

### Common Error Scenarios
1. **API Key Invalid**: Falls back to mock data
2. **Network Issues**: Falls back to mock data
3. **File Upload Failures**: Falls back to mock data
4. **API Rate Limits**: Falls back to mock data

## Usage Instructions

### 1. Set API Key
```bash
# Method 1: Environment variable
export GEMINI_API_KEY="your_api_key"

# Method 2: Start server with key
GEMINI_API_KEY="your_api_key" python3 food_detection_server.py
```

### 2. Start Server
```bash
python3 food_detection_server.py
```

### 3. Upload Files
- **Video**: Use video recorder to upload 10-second chunks
- **Audio**: Use audio recorder to upload 10-second chunks

### 4. View Results
- **Video Analysis**: Results appear in `#status_overlay4` and `#status_overlay2`
- **Audio Analysis**: Results appear in `#status_overlay3`

## Real-time Updates

### Streaming Integration
- Gemini analysis results are broadcast to all connected clients
- Real-time UI updates via SocketIO
- Multiple clients receive updates simultaneously

### Update Flow
```
File Upload → Gemini Analysis → HTML Generation → SocketIO Broadcast → UI Update
```

## Performance Considerations

### API Response Times
- **Video Analysis**: Typically 2-5 seconds
- **Audio Analysis**: Typically 1-3 seconds
- **File Upload**: Depends on file size

### Resource Management
- **File Cleanup**: Uploaded files deleted after analysis
- **Memory Management**: Async processing prevents blocking
- **Connection Pooling**: Efficient client reuse

## Security Considerations

### API Key Protection
- Environment variable storage
- No hardcoded keys in source code
- Server-side key management

### File Security
- Temporary file uploads
- Automatic cleanup after analysis
- No persistent storage of sensitive content

## Monitoring and Debugging

### Log Messages
```
[GEMINI] Client initialized successfully
[GEMINI] Analyzing video: /path/to/video.webm
[GEMINI] Video uploaded: video_file_name
[GEMINI] Video analysis completed: [analysis text]
[GEMINI] Cleaned up video file: video_file_name
```

### Error Logs
```
[GEMINI] Failed to initialize client: [error details]
[GEMINI] Error analyzing video: [error details]
```

## Testing

### Manual Testing
1. **API Test**: Use test script to verify Gemini connectivity
2. **Upload Test**: Upload test video/audio files
3. **UI Test**: Verify results appear in correct overlays
4. **Fallback Test**: Test with invalid API key

### Test Script
```python
#!/usr/bin/env python3
import os
from google import genai

api_key = os.getenv('GEMINI_API_KEY')
client = genai.Client(api_key=api_key, http_options={"base_url": "https://api.apiyi.com"})

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=["请用一句话介绍你自己。"]
)
print(response.text)
```

## Dependencies

### New Requirements
- `google-genai`: Google Gemini API client library

### Installation
```bash
pip install google-genai
```

## Cost Considerations

### API Usage
- **Gemini 2.5 Flash**: Charged per token
- **File Upload**: Charged per file
- **Analysis**: Charged per request

### Optimization
- **File Size Limits**: 10-second chunks minimize costs
- **Batch Processing**: Efficient API usage
- **Fallback Mode**: Reduces unnecessary API calls

## Future Enhancements

### Advanced Features
- **Custom Prompts**: Configurable analysis prompts
- **Multiple Models**: Support for different Gemini models
- **Batch Analysis**: Analyze multiple files simultaneously
- **Result Caching**: Cache analysis results

### Integration Options
- **Real-time Streaming**: Direct video stream analysis
- **Voice Commands**: Audio command recognition
- **Emotion Detection**: Advanced emotion analysis
- **Activity Classification**: Detailed activity categorization

## Troubleshooting

### Common Issues

1. **API Key Not Working**
   - Verify API key is valid
   - Check base URL configuration
   - Ensure proper environment variable setting

2. **Upload Failures**
   - Check file format compatibility
   - Verify file size limits
   - Check network connectivity

3. **Analysis Timeouts**
   - Increase timeout values
   - Check API rate limits
   - Verify file integrity

4. **UI Not Updating**
   - Check SocketIO connection
   - Verify client is connected
   - Check browser console for errors

### Debug Commands
```bash
# Check Gemini client status
curl -X GET http://localhost:5000/health

# Test API connectivity
python3 test_gemini.py

# Monitor server logs
tail -f server.log
```

## File Structure

### Modified Files
- `food_detection_server.py` - Gemini integration
- `requirements.txt` - Added google-genai dependency

### New Files
- `GEMINI_INTEGRATION_README.md` - This documentation

## API Reference

### Gemini Client Methods
```python
# Initialize client
client = genai.Client(api_key=api_key, http_options={"base_url": "https://api.apiyi.com"})

# Upload file
file = client.files.upload(path='path/to/file')

# Generate content
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[prompt, file]
)

# Delete file
client.files.delete(file.name)
```

### Response Format
```python
response.text  # String containing analysis results
```

This Gemini integration provides intelligent video and audio analysis capabilities, enhancing the camera AI demo with real-time AI-powered insights about user activities and environment.
