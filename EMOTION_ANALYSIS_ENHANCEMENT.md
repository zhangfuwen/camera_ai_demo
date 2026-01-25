# Enhanced Emotion and Facial Expression Analysis

This document describes the enhanced emotion and facial expression analysis capabilities using Gemini 3 Pro Preview model.

## Overview

The system has been upgraded to use **Gemini 3 Pro Preview** for more sophisticated emotion and facial expression analysis from both video and audio inputs. This provides deeper insights into user emotional states, facial expressions, and vocal characteristics.

## Enhanced Video Analysis

### Model Upgrade
- **Previous**: `gemini-2.5-flash`
- **Current**: `gemini-3-pro-preview`
- **Benefits**: Better emotion recognition, more detailed facial expression analysis

### Analysis Categories

#### 1. **面部表情分析** (Facial Expression Analysis)
- **主要情绪状态**: Primary emotion (快乐、悲伤、愤怒、惊讶、恐惧、厌恶、中性等)
- **情绪强度**: Emotion intensity (1-10 scale)
- **表情变化**: Facial expression changes description

#### 2. **肢体语言分析** (Body Language Analysis)
- **姿态和动作**: Posture and movement description
- **身体语言传达的情绪**: Emotions conveyed by body language
- **活动强度**: Activity intensity (低/中/高)

#### 3. **环境与活动** (Environment & Activity)
- **当前主要活动**: Current main activity
- **环境描述**: Environment description
- **检测到的关键对象**: Key detected objects

#### 4. **综合情绪评估** (Comprehensive Emotional Assessment)
- **整体心理状态**: Overall psychological state
- **注意力集中程度**: Attention concentration level
- **可能的情绪原因**: Possible emotional triggers

### Video Output Format

#### Status Overlay 4 (#status-overlay4)
```html
<div class="text-green-500">
    <div><strong>活动强度:</strong> 中</div>
    <div><strong>心理状态:</strong> 专注平静</div>
    <div><strong>情绪强度:</strong> 7/10</div>
    <div class="text-xs mt-1">分析时间: 14:30:25</div>
</div>
```

#### Status Overlay 2 (#status-overlay2)
```html
<div class="text-green-500">
    <div><strong>主要情绪:</strong> 专注</div>
    <div><strong>检测时间:</strong> 14:30:25</div>
    <div><strong>分析模型:</strong> Gemini-3-Pro</div>
</div>
```

## Enhanced Audio Analysis

### Model Upgrade
- **Previous**: `gemini-2.5-flash`
- **Current**: `gemini-3-pro-preview`
- **Benefits**: Better emotion detection from voice, improved transcription accuracy

### Analysis Categories

#### 1. **语音内容分析** (Speech Content Analysis)
- **完整转录文本**: Complete transcription text
- **语言和口音特征**: Language and accent features
- **语速和流畅度**: Speech speed and fluency

#### 2. **声音情绪分析** (Voice Emotion Analysis)
- **主要情绪状态**: Primary emotion (快乐、悲伤、愤怒、焦虑、兴奋、平静等)
- **情绪强度**: Emotion intensity (1-10 scale)
- **语调变化和情感色彩**: Tone changes and emotional coloration

#### 3. **声音特征** (Voice Characteristics)
- **音量水平**: Volume level (低/中/高)
- **音调高低**: Pitch level
- **声音清晰度**: Voice clarity

#### 4. **环境声音** (Environmental Sounds)
- **背景噪音描述**: Background noise description
- **环境音效**: Environmental sound effects
- **音频质量评估**: Audio quality assessment

#### 5. **综合心理状态** (Comprehensive Psychological State)
- **说话者心理状态推断**: Speaker's psychological state inference
- **情绪稳定性**: Emotional stability
- **可能的情绪触发因素**: Possible emotional triggers

### Audio Output Format

#### Status Overlay 3 (#status-overlay3)
```html
<div class="text-green-500">
    <div><strong>语音内容:</strong> 我正在测试这个系统...</div>
    <div><strong>情绪状态:</strong> 平静专注</div>
    <div><strong>音量:</strong> 中</div>
    <div><strong>音质:</strong> 清晰</div>
    <div class="text-xs mt-1">分析时间: 14:30:25</div>
</div>
```

## Technical Implementation

### File Upload Fix
- **Issue**: Gemini API file upload syntax errors
- **Solution**: Use file objects instead of file paths
- **Code**: 
```python
with open(file_path, 'rb') as file:
    uploaded_file = gemini_client.files.upload(file=file)
```

### Enhanced Parsing Logic
- **Structured Response Parsing**: Extract specific information from detailed Gemini responses
- **Key Information Extraction**: Parse emotion states, intensity levels, and activity data
- **Error Handling**: Robust fallback mechanisms for parsing failures

## Benefits of Gemini 3 Pro Preview

### Enhanced Capabilities
1. **Better Emotion Recognition**: More accurate facial expression and voice emotion detection
2. **Detailed Analysis**: Comprehensive breakdown of emotional states and expressions
3. **Improved Transcription**: Better speech-to-text accuracy and language understanding
4. **Contextual Understanding**: Better interpretation of emotional context and triggers

### Performance Characteristics
- **Analysis Time**: Slightly longer than gemini-2.5-flash due to deeper analysis
- **Accuracy**: Significantly improved emotion detection accuracy
- **Detail Level**: Much more detailed and structured responses

## Usage Instructions

### 1. Video Recording and Analysis
1. Record 10-second video chunks using the video recorder
2. Upload triggers automatic Gemini 3 Pro analysis
3. Results appear in:
   - **#status-overlay4**: Activity and psychological state
   - **#status-overlay2**: Primary emotion and detection info

### 2. Audio Recording and Analysis
1. Record 10-second audio chunks using the audio recorder
2. Upload triggers automatic Gemini 3 Pro analysis
3. Results appear in **#status-overlay3**: Speech content and emotional analysis

### 3. Real-time Updates
- All analysis results are broadcast to connected clients via SocketIO
- Multiple users receive updates simultaneously
- Real-time UI updates with detailed emotional insights

## Analysis Examples

### Video Analysis Example
```
[GEMINI] Video analysis completed:

1. **面部表情分析**：
   - 主要情绪状态：专注平静
   - 情绪强度：7/10
   - 表情变化：眉头微蹙显示专注，嘴角自然放松

2. **肢体语言分析**：
   - 姿态和动作描述：身体前倾，手指在键盘上快速移动
   - 身体语言传达的情绪：高度专注和投入
   - 活动强度：中

3. **环境与活动**：
   - 当前主要活动：电脑工作
   - 环境描述：办公环境，光线充足
   - 检测到的关键对象：电脑、键盘、显示器

4. **综合情绪评估**：
   - 整体心理状态：专注投入
   - 注意力集中程度：高度集中
   - 可能的情绪原因：工作任务驱动
```

### Audio Analysis Example
```
[GEMINI] Audio analysis completed:

1. **语音内容分析**：
   - 完整转录文本：我正在测试这个新的情绪分析系统，看看效果如何
   - 语言和口音特征：标准普通话，语速适中
   - 语速和流畅度：流畅自然，无明显停顿

2. **声音情绪分析**：
   - 主要情绪状态：平静好奇
   - 情绪强度：6/10
   - 语调变化和情感色彩：语调平稳，带有轻微的期待感

3. **声音特征**：
   - 音量水平：中
   - 音调高低：中等音调
   - 声音清晰度：清晰

4. **环境声音**：
   - 背景噪音描述：轻微的键盘敲击声
   - 环境音效：安静的室内环境
   - 音频质量评估：良好

5. **综合心理状态**：
   - 说话者心理状态推断：平静且带有探索性
   - 情绪稳定性：稳定
   - 可能的情绪触发因素：测试新系统的期待
```

## Configuration

### Model Selection
```python
# Video Analysis
model='gemini-3-pro-preview'

# Audio Analysis  
model='gemini-3-pro-preview'
```

### Prompt Engineering
- **Structured Prompts**: Detailed, sectioned prompts for comprehensive analysis
- **Chinese Language**: Optimized for Chinese emotional expression analysis
- **Specific Categories**: Clear categorization of analysis results

## Error Handling

### Fallback Mechanisms
- **Model Unavailable**: Falls back to mock data generation
- **Parse Failures**: Uses default values for missing information
- **Upload Errors**: Graceful degradation with error logging

### Logging
```
[GEMINI] Analyzing video: /path/to/video.webm
[GEMINI] Video uploaded: file_name
[GEMINI] Video analysis completed: [detailed analysis text]
[GEMINI] Cleaned up video file: file_name
```

## Performance Considerations

### Analysis Time
- **Video**: 3-6 seconds (depending on content complexity)
- **Audio**: 2-4 seconds (depending on speech content)
- **File Upload**: 1-2 seconds (depending on file size)

### Resource Usage
- **Memory**: Optimized file handling with automatic cleanup
- **Network**: Efficient file upload and API communication
- **Processing**: Asynchronous processing prevents blocking

## Future Enhancements

### Potential Improvements
1. **Real-time Analysis**: Stream processing for live emotion detection
2. **Multi-language Support**: Support for different languages and cultures
3. **Emotion History**: Track emotional state changes over time
4. **Custom Prompts**: User-configurable analysis parameters
5. **Integration**: Connect with other emotion analysis APIs

### Advanced Features
- **Emotion Trends**: Analyze emotional patterns over time
- **Multi-modal Analysis**: Combine video and audio for comprehensive analysis
- **Alert System**: Trigger alerts based on emotional state changes
- **Personalization**: Learn individual emotional patterns

## Troubleshooting

### Common Issues

1. **Analysis Time Too Long**
   - Check network connectivity
   - Verify file size limits
   - Monitor API rate limits

2. **Inaccurate Emotion Detection**
   - Ensure good lighting for video
   - Check audio quality for voice analysis
   - Verify proper file formats

3. **Parsing Errors**
   - Check Gemini response format changes
   - Verify prompt structure
   - Monitor structured response parsing

### Debug Commands
```bash
# Check server logs for Gemini analysis
tail -f /var/log/server.log | grep GEMINI

# Test API connectivity
python3 test_gemini.py

# Monitor file uploads
ls -la recorded_videos/ recorded_audio/
```

This enhanced emotion analysis system provides comprehensive insights into user emotional states through advanced AI-powered video and audio analysis, making it ideal for applications requiring deep emotional understanding and monitoring.
