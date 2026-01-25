from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS  # Add this import
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io
import base64
import json
import os
from datetime import datetime
import werkzeug.utils
import threading
import time
import random
import asyncio
from flask_socketio import SocketIO, emit
from openai import OpenAI

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# Global data storage for streaming
stream_data = {
    'sensor_values': {},
    'user_emotions': {},
    'audio_detections': [],
    'video_detections': [],
    'overall_status': {}
}

# Analysis results storage for summary generation
analysis_results = {
    'video_analysis': [],  # Store video analysis results
    'audio_analysis': [],  # Store audio analysis results
    'emotion_history': [], # Store emotion tracking data
    'activity_history': [] # Store activity tracking data
}

# Configuration for storage limits
MAX_STORED_RESULTS = 50  # Keep last 50 results of each type

# Connected clients
connected_clients = set()

# Gemini client configuration (using OpenAI library)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'sk-EplRjGkWQ9CwXK5w10936448E56a46BcB487EeE809C6Bd40')
gemini_client = None

try:
    gemini_client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://api.apiyi.com/v1"
    )
    print('[GEMINI] Client initialized successfully with OpenAI library')
except Exception as e:
    print(f'[GEMINI] Failed to initialize client: {e}')
    gemini_client = None

# Load the trained food detection model
# First try to load the trained model from the training directory
#model_path = "food_detection_training/runs/detect/food_det_model/weights/best.pt"
# model_path = "yolov8n-seg.pt"
model_path = "best.pt"
model_path = "./food_detection_training/runs/detect/food_det_model/weights/best.pt"
model_path = "./food_detection_training/yolov8n-seg.pt"
model_path = "yolo26x.pt"
fallback_model_path = "yolov8n.pt"

print("Loading food detection model...")
try:
    model = YOLO(model_path)
    print(f"Loaded trained model from {model_path}")
except Exception as e:
    print(f"Could not load trained model: {e}")
    print(f"Loading default YOLOv8 model instead...")
    model = YOLO(fallback_model_path)

# Get the model's class names
try:
    class_names = model.names  # This gets the class names from the trained model
    print(f"Model loaded with {len(class_names)} classes: {list(class_names.values())}")
except:
    class_names = {}  # Fallback if class names aren't available
    print("Could not retrieve class names from model")

# Create upload directory for videos if it doesn't exist
UPLOAD_FOLDER = 'recorded_videos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload directory: {UPLOAD_FOLDER}")

# Create upload directory for audio if it doesn't exist
AUDIO_UPLOAD_FOLDER = 'recorded_audio'
if not os.path.exists(AUDIO_UPLOAD_FOLDER):
    os.makedirs(AUDIO_UPLOAD_FOLDER)
    print(f"Created audio upload directory: {AUDIO_UPLOAD_FOLDER}")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['AUDIO_UPLOAD_FOLDER'] = AUDIO_UPLOAD_FOLDER



@app.route('/')
def home():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files like CSS, JS, images, etc."""
    return send_from_directory('.', path)

@app.route('/detect_food', methods=['POST'])
def detect_food():
    """
    Handle POST requests with base64 encoded images for YOLO segmentation-based food detection.
    
    Expected JSON format:
    {
        "image": "base64_encoded_image_string"
    }
    
    Returns:
    {
        "success": true,
        "detections": [
            {
                "label": "food_item_name",
                "score": confidence_score,
                "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                "mask": [[0, 0, 1, 1, ...], ...]  // 2D array representing the segmentation mask
            }
        ]
    }
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data provided"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
      #  img = img.transpose(1, 0, 2)
        
        if img is None:
            return jsonify({"success": False, "error": "Could not decode image"}), 400

        print("Image shape:", img.shape)
        
        # Perform segmentation with YOLO
        results = model(img, conf=0.1)  # Confidence threshold of 0.5
        
        detections = []
        
        # Process detections
        if results and len(results) > 0:
            r = results[0]
            
            # Get boxes, scores, and class IDs
            boxes = r.boxes.xyxy.cpu().numpy()  # Bounding boxes
            scores = r.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = r.boxes.cls.cpu().numpy()  # Class IDs
            
            # Get masks if available
            masks = r.masks.data.cpu().numpy() if r.masks is not None else None
            
            for i in range(len(boxes)):
                box = boxes[i]
                score = float(scores[i])
                class_id = int(class_ids[i])
                
                # Convert class ID to label - all models have their class names embedded
                label = model.names.get(class_id, f"Class {class_id}")
                
                # Include all detections regardless of class
                # Prepare detection data
                detection = {
                    "label": label,
                    "score": score,
                    "box": {
                        "xmin": float(box[0]),
                        "ymin": float(box[1]),
                        "xmax": float(box[2]),
                        "ymax": float(box[3])
                    }
                }
                
                # Add mask data if available
                if masks is not None and i < len(masks):
                    # Convert mask tensor to a 2D array of 0s and 1s
                    mask_tensor = masks[i]
                    # Convert to binary mask and then to list format
                    mask_array = (mask_tensor > 0.5).astype(int).tolist()
                    detection["mask"] = mask_array
                
                detections.append(detection)
        
        return jsonify({
            "success": True,
            "detections": detections
        })
    
    except Exception as e:
        app.logger.error(f"Error in detect_food: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/detect_food_stream', methods=['POST'])
def detect_food_stream():
    """
    Handle POST requests with base64 encoded images for YOLO segmentation-based food detection.
    Same functionality as /detect_food but with a different endpoint name.
    
    Expected JSON format:
    {
        "image": "base64_encoded_image_string"
    }
    
    Returns:
    {
        "success": true,
        "detections": [
            {
                "label": "food_item_name",
                "score": confidence_score,
                "box": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                "mask": [[0, 0, 1, 1, ...], ...]  // 2D array representing the segmentation mask
            }
        ]
    }
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data provided"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"success": False, "error": "Could not decode image"}), 400
        
        # Perform segmentation with YOLO
        results = model(img, conf=0.5)  # Confidence threshold of 0.5
        
        detections = []
        
        # Process detections
        if results and len(results) > 0:
            r = results[0]
            
            # Get boxes, scores, and class IDs
            boxes = r.boxes.xyxy.cpu().numpy()  # Bounding boxes
            scores = r.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = r.boxes.cls.cpu().numpy()  # Class IDs
            
            # Get masks if available
            masks = r.masks.data.cpu().numpy() if r.masks is not None else None
            
            for i in range(len(boxes)):
                box = boxes[i]
                score = float(scores[i])
                class_id = int(class_ids[i])
                
                # Convert class ID to label - all models have their class names embedded
                label = model.names.get(class_id, f"Class {class_id}")
                
                # Include all detections regardless of class
                # Prepare detection data
                detection = {
                    "label": label,
                    "score": score,
                    "box": {
                        "xmin": float(box[0]),
                        "ymin": float(box[1]),
                        "xmax": float(box[2]),
                        "ymax": float(box[3])
                    }
                }
                
                # Add mask data if available
                if masks is not None and i < len(masks):
                    # Convert mask tensor to a 2D array of 0s and 1s
                    mask_tensor = masks[i]
                    # Convert to binary mask and then to list format
                    mask_array = (mask_tensor > 0.5).astype(int).tolist()
                    detection["mask"] = mask_array
                
                detections.append(detection)
        
        return jsonify({
            "success": True,
            "detections": detections
        })
    
    except Exception as e:
        app.logger.error(f"Error in detect_food_stream: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/upload_video', methods=['POST'])
def upload_video():
    """
    Handle video file uploads and save them with timestamps.
    
    Expected form data:
    - video: video file (webm format)
    - timestamp: timestamp string from client
    
    Returns:
    {
        "success": true,
        "filename": "saved_filename.webm",
        "timestamp": "timestamp_string",
        "path": "path/to/saved/file"
    }
    """
    try:
        # Log request received
        print(f"[VIDEO UPLOAD] Request received at {datetime.now().isoformat()}")
        print(f"[VIDEO UPLOAD] Request headers: {dict(request.headers)}")
        print(f"[VIDEO UPLOAD] Request form data keys: {list(request.form.keys())}")
        print(f"[VIDEO UPLOAD] Request files keys: {list(request.files.keys())}")
        
        # Check if video file is in the request
        if 'video' not in request.files:
            print(f"[VIDEO UPLOAD] ERROR: No video file in request")
            return jsonify({"success": False, "error": "No video file provided"}), 400
        
        video_file = request.files['video']
        print(f"[VIDEO UPLOAD] Video file received: {video_file.filename}")
        print(f"[VIDEO UPLOAD] Video file content type: {video_file.content_type}")
        
        if video_file.filename == '':
            print(f"[VIDEO UPLOAD] ERROR: Empty filename")
            return jsonify({"success": False, "error": "No video file selected"}), 400
        
        # Get timestamp from form data or generate one
        timestamp = request.form.get('timestamp', datetime.now().isoformat())
        print(f"[VIDEO UPLOAD] Client timestamp: {timestamp}")
        
        # Secure the filename and ensure it has the correct extension
        filename = werkzeug.utils.secure_filename(video_file.filename)
        # if not filename.lower().endswith('.webm'):
        #     filename += '.webm'
        
        # Add timestamp prefix to ensure unique ordering
        timestamp_prefix = timestamp.replace(':', '-').replace('.', '-')
        saved_filename = f"{timestamp_prefix}_{filename}"
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        print(f"[VIDEO UPLOAD] Saving file to: {file_path}")
        
        video_file.save(file_path)
        
        # Get file size for logging
        file_size = os.path.getsize(file_path)
        print(f"[VIDEO UPLOAD] File saved successfully: {saved_filename}")
        print(f"[VIDEO UPLOAD] File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        print(f"[VIDEO UPLOAD] Full path: {os.path.abspath(file_path)}")
        
        # List files in upload directory for verification
        try:
            existing_files = os.listdir(app.config['UPLOAD_FOLDER'])
            print(f"[VIDEO UPLOAD] Total files in upload directory: {len(existing_files)}")
            if len(existing_files) <= 5:  # Only list if not too many files
                print(f"[VIDEO UPLOAD] Files: {sorted(existing_files)}")
        except Exception as e:
            print(f"[VIDEO UPLOAD] Warning: Could not list directory contents: {e}")
        
        print(f"[VIDEO UPLOAD] Upload completed successfully at {datetime.now().isoformat()}")
        
        # Trigger Gemini video analysis in background
        gemini_thread = threading.Thread(target=run_gemini_video_analysis, args=(file_path,), daemon=True)
        gemini_thread.start()
        print(f"[GEMINI] Started video analysis for: {saved_filename}")
        
        return jsonify({
            "success": True,
            "filename": saved_filename,
            "timestamp": timestamp,
            "path": file_path,
            "size": file_size
        })
    
    except Exception as e:
        print(f"[VIDEO UPLOAD] ERROR: {str(e)}")
        print(f"[VIDEO UPLOAD] Error type: {type(e).__name__}")
        import traceback
        print(f"[VIDEO UPLOAD] Traceback: {traceback.format_exc()}")
        app.logger.error(f"Error in upload_video: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """
    Handle audio file uploads and save them with timestamps.
    
    Expected form data:
    - audio: audio file (webm format)
    - timestamp: timestamp string from client
    
    Returns:
    {
        "success": true,
        "filename": "saved_filename.webm",
        "timestamp": "timestamp_string",
        "path": "path/to/saved/file"
    }
    """
    try:
        # Log request received
        print(f"[AUDIO UPLOAD] Request received at {datetime.now().isoformat()}")
        print(f"[AUDIO UPLOAD] Request headers: {dict(request.headers)}")
        print(f"[AUDIO UPLOAD] Request form data keys: {list(request.form.keys())}")
        print(f"[AUDIO UPLOAD] Request files keys: {list(request.files.keys())}")
        
        # Check if audio file is in the request
        if 'audio' not in request.files:
            print(f"[AUDIO UPLOAD] ERROR: No audio file in request")
            return jsonify({"success": False, "error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        print(f"[AUDIO UPLOAD] Audio file received: {audio_file.filename}")
        print(f"[AUDIO UPLOAD] Audio file content type: {audio_file.content_type}")
        
        if audio_file.filename == '':
            print(f"[AUDIO UPLOAD] ERROR: Empty filename")
            return jsonify({"success": False, "error": "No audio file selected"}), 400
        
        # Get timestamp from form data or generate one
        timestamp = request.form.get('timestamp', datetime.now().isoformat())
        print(f"[AUDIO UPLOAD] Client timestamp: {timestamp}")
        
        # Secure the filename and ensure it has the correct extension
        filename = werkzeug.utils.secure_filename(audio_file.filename)
        if not filename.lower().endswith('.webm'):
            filename += '.webm'
        
        # Add timestamp prefix to ensure unique ordering
        timestamp_prefix = timestamp.replace(':', '-').replace('.', '-')
        saved_filename = f"{timestamp_prefix}_{filename}"
        
        # Save the file
        file_path = os.path.join(app.config['AUDIO_UPLOAD_FOLDER'], saved_filename)
        print(f"[AUDIO UPLOAD] Saving file to: {file_path}")
        
        audio_file.save(file_path)
        
        # Get file size for logging
        file_size = os.path.getsize(file_path)
        print(f"[AUDIO UPLOAD] File saved successfully: {saved_filename}")
        print(f"[AUDIO UPLOAD] File size: {file_size} bytes ({file_size / 1024:.2f} KB)")
        print(f"[AUDIO UPLOAD] Full path: {os.path.abspath(file_path)}")
        
        # List files in upload directory for verification
        try:
            existing_files = os.listdir(app.config['AUDIO_UPLOAD_FOLDER'])
            print(f"[AUDIO UPLOAD] Total files in audio upload directory: {len(existing_files)}")
            if len(existing_files) <= 5:  # Only list if not too many files
                print(f"[AUDIO UPLOAD] Files: {sorted(existing_files)}")
        except Exception as e:
            print(f"[AUDIO UPLOAD] Warning: Could not list directory contents: {e}")
        
        print(f"[AUDIO UPLOAD] Upload completed successfully at {datetime.now().isoformat()}")
        
        # Trigger Gemini audio analysis in background
        gemini_thread = threading.Thread(target=run_gemini_audio_analysis, args=(file_path,), daemon=True)
        gemini_thread.start()
        print(f"[GEMINI] Started audio analysis for: {saved_filename}")
        
        return jsonify({
            "success": True,
            "filename": saved_filename,
            "timestamp": timestamp,
            "path": file_path,
            "size": file_size
        })
    
    except Exception as e:
        print(f"[AUDIO UPLOAD] ERROR: {str(e)}")
        print(f"[AUDIO UPLOAD] Error type: {type(e).__name__}")
        import traceback
        print(f"[AUDIO UPLOAD] Traceback: {traceback.format_exc()}")
        app.logger.error(f"Error in upload_audio: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})


@app.route('/classes', methods=['GET'])
def get_classes():
    """Return the list of available food classes"""
    try:
        class_names = list(model.names.values()) if hasattr(model, 'names') else []
        return jsonify({
            'success': True,
            'classes': class_names,
            'count': len(class_names)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# WebSocket events
@socketio.on('connect')
def handle_connect():
    print(f'[STREAM] Client connected: {request.sid}')
    connected_clients.add(request.sid)
    emit('status', {'message': 'Connected to stream'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'[STREAM] Client disconnected: {request.sid}')
    connected_clients.discard(request.sid)

@socketio.on('heartbeat')
def handle_heartbeat():
    emit('heartbeat_response', {'timestamp': datetime.now().isoformat()})

def broadcast_to_clients(message_type, content):
    """Broadcast message to all connected clients"""
    if connected_clients:
        message = {
            'type': message_type,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        socketio.emit('stream_update', message)
        print(f'[STREAM] Broadcasted {message_type} to {len(connected_clients)} clients')

# Global variables for realistic sensor data simulation
sensor_state = {
    'energy': 75,
    'heartRate': 72,
    'blood_pressure': 85,
    'steps': 3450,
    'calories': 425,
    'sleep_hours': 7.23
}

def generate_sensor_data():
    """Generate realistic sensor data like from a sport watch with smooth changes"""
    global sensor_state
    
    # Energy level (20-100): 保持不变
    # energy_change = random.uniform(-2, 2)
    # sensor_state['energy'] = max(20, min(100, sensor_state['energy'] + energy_change))
    
    # Heart rate (60-100): fluctuates around baseline with small changes
    heart_rate_change = random.uniform(-3, 3)
    sensor_state['heartRate'] = max(60, min(100, sensor_state['heartRate'] + heart_rate_change))
    
    # Blood pressure (70-120): very stable, minimal changes
    bp_change = random.uniform(-1, 1)
    sensor_state['blood_pressure'] = max(70, min(120, sensor_state['blood_pressure'] + bp_change))
    
    # Steps: 保持不变
    # steps_increment = random.randint(0, 5)  # 0-5 steps per second when active
    # sensor_state['steps'] += steps_increment
    
    # Calories: gradually increases based on activity
    calorie_increment = random.randint(0, 2)  # 0-2 calories per second
    sensor_state['calories'] += calorie_increment
    
    # Sleep hours: remains constant during the day
    # Only change this occasionally to simulate different days
    if random.random() < 0.001:  # Very small chance to change
        sensor_state['sleep_hours'] = round(random.uniform(6.5, 8.5), 2)
    
    return {
        'energy': int(sensor_state['energy']),
        'sleep': f"{sensor_state['sleep_hours']:.2f} Hours",
        'sport': f"{sensor_state['steps']} Steps",
        'blood': f"{int(sensor_state['blood_pressure'])}",
        'heartRate': f"{int(sensor_state['heartRate'])}",
        'calories': f"{sensor_state['calories']}"
    }

def generate_sensor_html():
    """Generate plain text for sensor values"""
    data = generate_sensor_data()
    return f"""能量值: {data['energy']}
睡眠时长: {data['sleep']} 小时
运动步数: {data['sport']} 步
血压: {data['blood']}
心率: {data['heartRate']} 次/分
卡路里: {data['calories']} 卡"""

def generate_emotion_html():
    """Generate plain text for user emotions from stored analysis results"""
    if analysis_results['emotion_history']:
        # Get the most recent emotion analysis
        latest_emotion = analysis_results['emotion_history'][-1]
        return f"""情绪状态: {latest_emotion['emotion']}
强度: {latest_emotion['intensity']}
检测时间: {datetime.fromisoformat(latest_emotion['timestamp']).strftime('%H:%M:%S')}"""
    else:
        print("[EMOTION] No emotion analysis results available", analysis_results['emotion_history'])
        # Fallback if no analysis results available
        return "情绪状态: 等待分析数据..."

def generate_audio_detection_html():
    """Generate plain text for audio detection results from stored analysis results"""
    if analysis_results['audio_analysis']:
        # Get the most recent audio analysis
        latest_audio = analysis_results['audio_analysis'][-1]
        return f"""语音内容: {latest_audio['speech_content']}
情绪状态: {latest_audio['emotion_state']}
音量: {latest_audio['volume_level']}
音质: {latest_audio['voice_quality']}
分析时间: {datetime.fromisoformat(latest_audio['timestamp']).strftime('%H:%M:%S')}"""
    else:
        print("[AUDIO] No audio analysis results available", analysis_results['audio_analysis'])
        # Fallback if no analysis results available
        return "音频分析: 等待分析数据..."

def generate_video_detection_html():
    """Generate plain text for video detection results from stored analysis results"""
    if analysis_results['video_analysis']:
        # Get the most recent video analysis
        latest_video = analysis_results['video_analysis'][-1]
        return f"""活动强度: {latest_video['activity_level']}
心理状态: {latest_video['overall_state']}
情绪强度: {latest_video['emotion_intensity']}
主要情绪: {latest_video['emotion_state']}
分析时间: {datetime.fromisoformat(latest_video['timestamp']).strftime('%H:%M:%S')}"""
    else:
        # Fallback if no analysis results available
        return "视频分析: 等待分析数据..."

def generate_overall_status_html():
    """Generate plain text for overall user status from stored analysis results"""
    total_analyses = len(analysis_results['video_analysis']) + len(analysis_results['audio_analysis'])
    
    if total_analyses > 0:
        # Get recent analyses for summary
        recent_emotions = [e['emotion'] for e in analysis_results['emotion_history'][-5:]]
        recent_activities = [a['activity'] for a in analysis_results['activity_history'][-5:]]
        
        # Determine most common emotion and activity
        if recent_emotions:
            most_common_emotion = max(set(recent_emotions), key=recent_emotions.count)
        else:
            most_common_emotion = "未知"
            
        if recent_activities:
            most_common_activity = max(set(recent_activities), key=recent_activities.count)
        else:
            most_common_activity = "未知"
        
        return f"""整体状态: 基于最近{total_analyses}次分析
主要情绪: {most_common_emotion}
主要活动: {most_common_activity}
数据更新: {datetime.now().strftime('%H:%M:%S')}
分析次数: 视频{len(analysis_results['video_analysis'])}次, 音频{len(analysis_results['audio_analysis'])}次"""
    else:
        # Fallback if no analysis results available
        return "整体状态: 等待分析数据..."

def sensor_update_timer():
    """Timer for updating sensor values every 1 second"""
    while True:
        try:
            html_content = generate_sensor_html()
            broadcast_to_clients('sensor_update', html_content)
            time.sleep(1)
        except Exception as e:
            print(f'[STREAM] Error in sensor timer: {e}')
            time.sleep(1)

def emotion_update_timer():
    """Timer for updating emotions every 3 seconds"""
    while True:
        try:
            html_content = generate_emotion_html()
            broadcast_to_clients('emotion_update', html_content)
            time.sleep(3)
        except Exception as e:
            print(f'[STREAM] Error in emotion timer: {e}')
            time.sleep(3)

def overall_status_timer():
    """Timer for updating overall status every 20 seconds"""
    while True:
        try:
            # Generate comprehensive summary using stored analysis results
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                summary_text = loop.run_until_complete(generate_comprehensive_summary())
                summary_html = generate_summary_html(summary_text)
                broadcast_to_clients('overall_status', summary_html)
            finally:
                loop.close()
            time.sleep(20)
        except Exception as e:
            print(f'[STREAM] Error in overall status timer: {e}')
            time.sleep(20)

def comprehensive_summary_timer():
    """Timer for generating comprehensive summaries every 5 minutes"""
    while True:
        try:
            print('[SUMMARY] Starting comprehensive summary generation...')
            # Generate comprehensive summary using stored analysis results
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                summary_text = loop.run_until_complete(generate_comprehensive_summary())
                summary_html = generate_summary_html(summary_text)
                broadcast_to_clients('overall_status', summary_html)
                print('[SUMMARY] Comprehensive summary generated and broadcasted')
            finally:
                loop.close()
            time.sleep(300)  # 5 minutes
        except Exception as e:
            print(f'[SUMMARY] Error in comprehensive summary timer: {e}')
            time.sleep(300)

async def analyze_video_with_gemini(video_path):
    """Analyze video using Gemini model"""
    try:
        if not gemini_client:
            print('[GEMINI] Client not available, using fallback')
            # Fallback to mock data
            html_content = generate_video_detection_html()
            broadcast_to_clients('video_detect', html_content)
            emotion_html = generate_emotion_html()
            broadcast_to_clients('emotion_update', emotion_html)
            return
        
        print(f'[GEMINI] Analyzing video: {video_path}')
        
        # Check if file exists and get file info
        import os
        if not os.path.exists(video_path):
            print(f'[GEMINI] Error: Video file does not exist: {video_path}')
            return
        
        file_size = os.path.getsize(video_path)
        print(f'[GEMINI] Video file info - Path: {video_path}, Size: {file_size} bytes')
        
        # Test Gemini client connectivity
        try:
            print(f'[GEMINI] Testing client connectivity...')
            test_response = gemini_client.chat.completions.create(
                model='gemini-3-pro-preview',
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=50
            )
            print(f'[GEMINI] Client connectivity test passed')
        except Exception as e:
            print(f'[GEMINI] Client connectivity test failed: {e}')
            return
        
        # Read and encode video file
        try:
            with open(video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode()
                video_url = f"data:video/mp4;base64,{video_b64}"
            print(f'[GEMINI] Video encoded successfully')
        except Exception as e:
            print(f'[GEMINI] Error encoding video: {e}')
            return
        
        # Analyze video content
        analysis_prompt = """请详细分析这个视频中的人物，并以JSON格式返回结果。

请返回以下结构的JSON：
{
    "emotion_state": "主要情绪状态（快乐、悲伤、愤怒、惊讶、恐惧、厌恶、中性等）",
    "emotion_intensity": "情绪强度（1-10分）",
    "facial_expression": "面部表情变化描述",
    "body_language": "姿态和动作描述",
    "body_emotion": "身体语言传达的情绪",
    "activity_level": "活动强度（低/中/高）",
    "current_activity": "当前主要活动",
    "environment_description": "环境描述",
    "diet_status": "饮食状况与健康状况",
    "overall_mental_state": "整体心理状态",
    "attention_level": "注意力集中程度（高/中/低）",
    "emotion_cause": "可能的情绪原因"
}

请确保返回的是有效的JSON格式，不要包含任何其他文本或说明。"""
        
        response = gemini_client.chat.completions.create(
            model="gemini-3-pro-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in video analysis. Always return valid JSON format."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": video_url
                            },
                            "mime_type": "video/mp4",
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=4096
        )
        
        analysis_text = response.choices[0].message.content
        print(f'[GEMINI] Video analysis completed successfully')
        
        # Parse JSON response
        try:
            # Try to extract JSON from the response
            import json
            import re
            
            # First, try to extract JSON from markdown code blocks
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, analysis_text, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                # Extract JSON from markdown code block
                json_str = json_match.group(1)
                print(f'[GEMINI] Found JSON in markdown code block')
            else:
                # Try to find JSON object in the response (without markdown)
                json_pattern = r'\{.*\}'
                json_match = re.search(json_pattern, analysis_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    print(f'[GEMINI] Found JSON object in response')
                else:
                    # Fallback: try to parse the entire response as JSON
                    json_str = analysis_text.strip()
                    print(f'[GEMINI] Parsing entire response as JSON')
            
            # Clean up the JSON string
            json_str = json_str.strip()
            
            # Parse the JSON
            video_analysis = json.loads(json_str)
            
            # Extract values from JSON with defaults
            emotion_state = video_analysis.get('emotion_state', '未检测到')
            emotion_intensity = video_analysis.get('emotion_intensity', '未知')
            activity_level = video_analysis.get('activity_level', '未知')
            overall_state = video_analysis.get('overall_mental_state', '正常')
            
            print(f'[GEMINI] Successfully parsed video analysis JSON: {emotion_state}, {activity_level}')
            
        except Exception as e:
            print(f'[GEMINI] Error parsing video analysis JSON: {e}')
            print(f'[GEMINI] Raw response: {analysis_text}')
            
            # Fallback values if JSON parsing fails
            emotion_state = "解析失败"
            emotion_intensity = "未知"
            activity_level = "未知"
            overall_state = "解析错误"
        
        # Generate comprehensive video detection text
        video_text = f"活动强度: {activity_level}\n心理状态: {overall_state}\n情绪强度: {emotion_intensity}\n分析时间: {datetime.now().strftime('%H:%M:%S')}"
        broadcast_to_clients('video_detect', video_text)
        
        # Generate detailed emotion text
        emotion_text = f"主要情绪: {emotion_state}\n检测时间: {datetime.now().strftime('%H:%M:%S')}\n分析模型: Gemini-3-Pro"
        broadcast_to_clients('emotion_update', emotion_text)
        
        # Store analysis results for summary generation
        video_result = {
            'timestamp': datetime.now().isoformat(),
            'emotion_state': emotion_state,
            'emotion_intensity': emotion_intensity,
            'activity_level': activity_level,
            'overall_state': overall_state,
            'full_analysis': analysis_text,
            'file_path': video_path
        }
        
        # Add to storage with size limit
        analysis_results['video_analysis'].append(video_result)
        print(f'[STORAGE] Stored video analysis result: {analysis_results["video_analysis"][-1]}')
        if len(analysis_results['video_analysis']) > MAX_STORED_RESULTS:
            analysis_results['video_analysis'].pop(0)
        
        # Store emotion history
        emotion_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': 'video',
            'emotion': emotion_state,
            'intensity': emotion_intensity,
            'confidence': 'high'
        }
        analysis_results['emotion_history'].append(emotion_entry)
        if len(analysis_results['emotion_history']) > MAX_STORED_RESULTS:
            analysis_results['emotion_history'].pop(0)
        
        # Store activity history
        activity_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': 'video',
            'activity': overall_state,
            'intensity': activity_level
        }
        analysis_results['activity_history'].append(activity_entry)
        if len(analysis_results['activity_history']) > MAX_STORED_RESULTS:
            analysis_results['activity_history'].pop(0)
        
        print(f'[STORAGE] Stored video analysis results. Total stored: {len(analysis_results["video_analysis"])}')
        
        # Clean up uploaded file
        try:
            gemini_client.files.delete(uploaded_file.name)
            print(f'[GEMINI] Cleaned up video file: {uploaded_file.name}')
        except:
            pass
        
        print(f'[GEMINI] Video analysis completed for: {video_path}')
        
    except Exception as e:
        print(f'[GEMINI] Error analyzing video: {e}')
        # Fallback to mock data
        html_content = generate_video_detection_html()
        broadcast_to_clients('video_detect', html_content)
        emotion_html = generate_emotion_html()
        broadcast_to_clients('emotion_update', emotion_html)

async def generate_comprehensive_summary():
    """Generate comprehensive summary using stored analysis results"""
    try:
        if not gemini_client:
            print('[SUMMARY] Gemini client not available, using fallback')
            return generate_mock_summary()
        
        # Check if we have enough data for meaningful summary
        total_analyses = len(analysis_results['video_analysis']) + len(analysis_results['audio_analysis'])
        if total_analyses < 2:
            print('[SUMMARY] Insufficient data for comprehensive summary')
            return generate_mock_summary()
        
        print(f'[SUMMARY] Generating comprehensive summary from {total_analyses} analyses')
        
        # Prepare data for summary
        recent_video = analysis_results['video_analysis'][-5:]  # Last 5 video analyses
        recent_audio = analysis_results['audio_analysis'][-5:]  # Last 5 audio analyses
        emotion_history = analysis_results['emotion_history'][-10:]  # Last 10 emotion entries
        activity_history = analysis_results['activity_history'][-10:]  # Last 10 activity entries
        
        # Check for eating activities in recent analyses
        eating_activities = []
        for activity in activity_history:
            if any(keyword in activity['activity'].lower() for keyword in ['吃', '进食', '饮食', '食物', 'eating', 'meal', 'food']):
                eating_activities.append(activity)
        
        # Check for diet-related content in video analyses
        diet_related_videos = []
        for video in recent_video:
            if any(keyword in str(video.get('full_analysis', '')).lower() for keyword in ['吃', '进食', '饮食', '食物', 'eating', 'meal', 'food', 'diet']):
                diet_related_videos.append(video)
        
        has_eating_activity = len(eating_activities) > 0 or len(diet_related_videos) > 0
        
        # Build summary prompt
        summary_prompt = f"""基于以下分析结果，请生成一个综合性的用户状态总结：

## 视频分析结果（最近{len(recent_video)}次）：
{chr(10).join([f"- {v['timestamp']}: 情绪={v['emotion_state']}, 强度={v['emotion_intensity']}, 状态={v['overall_state']}" for v in recent_video])}

## 音频分析结果（最近{len(recent_audio)}次）：
{chr(10).join([f"- {a['timestamp']}: 情绪={a['emotion_state']}, 音量={a['volume_level']}, 内容={a['speech_content']}" for a in recent_audio])}

## 情绪历史（最近{len(emotion_history)}次）：
{chr(10).join([f"- {e['timestamp']}: {e['source']} - {e['emotion']} (强度: {e['intensity']})" for e in emotion_history])}

## 活动历史（最近{len(activity_history)}次）：
{chr(10).join([f"- {a['timestamp']}: {a['source']} - {a['activity']}" for a in activity_history])}

## 进食活动检测：
{f"检测到 {len(eating_activities)} 次进食相关活动" if has_eating_activity else "未检测到进食活动"}

请基于以上数据，提供以下总结：

1. **整体情绪趋势分析**：
   - 主要情绪模式
   - 情绪变化趋势
   - 情绪稳定性评估

2. **活动模式分析**：
   - 主要活动类型
   - 活动强度变化
   - 工作/休息平衡

3. **沟通模式分析**：
   - 语音活动频率
   - 沟通情绪特征
   - 表达方式特点

4. **综合状态评估**：
   - 当前整体状态
   - 压力水平评估
   - 效率状态评估

5. **建议和提醒**：
   - 健康建议{'（重点关注饮食健康和营养均衡）' if has_eating_activity else ''}
   - 工作效率建议
   - 情绪管理建议

{'''6. **饮食健康专项建议**：
   - 进食时间规律性建议
   - 营养搭配建议
   - 健康饮食习惯建议
   - 消化健康提醒''' if has_eating_activity else ''}

请用结构化的中文回答，每个部分详细分析并提供具体建议。每个部分只输出一句简短描述。{'如果检测到进食活动，请在健康建议部分特别关注饮食健康和营养均衡。' if has_eating_activity else ''}"""

        # Generate summary using Gemini
        response = gemini_client.chat.completions.create(
            model="gemini-3-pro-preview",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3,
            max_tokens=4096
        )
        
        summary_text = response.choices[0].message.content
        print(f'[SUMMARY] Comprehensive summary generated successfully')
        
        return summary_text
        
    except Exception as e:
        print(f'[SUMMARY] Error generating comprehensive summary: {e}')
        return generate_mock_summary()

def generate_mock_summary():
    """Generate mock summary when Gemini is unavailable"""
    import random
    
    emotion_states = ['专注', '平静', '积极', '放松', '投入']
    activity_levels = ['中等', '较高', '适中', '稳定']
    
    # Check for eating activities in recent analyses
    eating_activities = []
    activity_history = analysis_results.get('activity_history', [])
    for activity in activity_history[-10:]:  # Last 10 activities
        if any(keyword in activity['activity'].lower() for keyword in ['吃', '进食', '饮食', '食物', 'eating', 'meal', 'food']):
            eating_activities.append(activity)
    
    has_eating_activity = len(eating_activities) > 0
    
    # Generate base summary
    mock_summary = f"""
## 综合状态总结

### 1. 整体情绪趋势分析
- **主要情绪模式**: {random.choice(emotion_states)}
- **情绪变化趋势**: 情绪状态相对稳定
- **情绪稳定性评估**: 良好

### 2. 活动模式分析
- **主要活动类型**: 电脑工作
- **活动强度变化**: {random.choice(activity_levels)}
- **工作/休息平衡**: 需要注意适当休息

### 3. 沟通模式分析
- **语音活动频率**: 适中
- **沟通情绪特征**: 积极正面
- **表达方式特点**: 清晰流畅

### 4. 综合状态评估
- **当前整体状态**: 良好
- **压力水平评估**: 中等
- **效率状态评估**: 高效

### 5. 建议和提醒
- **健康建议**: {'定时休息，注意饮食营养均衡，避免暴饮暴食，建议细嚼慢咽有助于消化健康' if has_eating_activity else '定时休息，保护视力'}
- **工作效率建议**: 保持专注，适当调整
- **情绪管理建议**: 保持积极心态"""
    
    # Add eating-specific advice if eating activity detected
    if has_eating_activity:
        mock_summary += f"""

### 6. 饮食健康专项建议
- **进食时间规律性建议**: 建议固定用餐时间，避免不规律进食
- **营养搭配建议**: 注意荤素搭配，保证营养均衡
- **健康饮食习惯建议**: 细嚼慢咽，避免边工作边进食
- **消化健康提醒**: 饭后适当休息，有助于消化吸收

## 进食活动检测
检测到 {len(eating_activities)} 次进食相关活动，建议关注饮食健康和营养均衡。"""
    
    mock_summary += """

*注：此为模拟总结，实际分析需要更多数据*
"""
    
    return mock_summary

def generate_summary_html(summary_text):
    """Generate plain text for summary display"""
    # Convert markdown-style headers to plain text
    import re
    
    # Convert headers to plain text with separators
    summary_text = re.sub(r'^### (.*?)$', r'---\n\1\n---', summary_text, flags=re.MULTILINE)
    summary_text = re.sub(r'^## (.*?)$', r'===\n\1\n===', summary_text, flags=re.MULTILINE)
    summary_text = re.sub(r'^# (.*?)$', r'***\n\1\n***', summary_text, flags=re.MULTILINE)
    
    # Convert bold text to plain text
    summary_text = re.sub(r'\*\*(.*?)\*\*', r'\1', summary_text)
    
    # Convert list items to plain text with bullets
    summary_text = re.sub(r'^- (.*?)$', r'• \1', summary_text, flags=re.MULTILINE)
    
    # Clean up extra line breaks
    summary_text = re.sub(r'\n{3,}', '\n\n', summary_text)
    
    # Wrap in container with simple format
    text_content = f"""📊 综合状态总结
生成时间: {datetime.now().strftime('%H:%M:%S')}
{summary_text}"""
    
    return text_content

async def analyze_audio_with_gemini(audio_path):
    """Analyze audio using Gemini model"""
    try:
        if not gemini_client:
            print('[GEMINI] Client not available, using fallback')
            # Fallback to mock data
            html_content = generate_audio_detection_html()
            broadcast_to_clients('audio_detect', html_content)
            return
        
        print(f'[GEMINI] Analyzing audio: {audio_path}')
        
        # Check if file exists and get file info
        import os
        if not os.path.exists(audio_path):
            print(f'[GEMINI] Error: Audio file does not exist: {audio_path}')
            return
        
        file_size = os.path.getsize(audio_path)
        print(f'[GEMINI] Audio file info - Path: {audio_path}, Size: {file_size} bytes')
        
        # Read and encode audio file
        try:
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
                audio_url = f"data:audio/webm;base64,{audio_b64}"
            print(f'[GEMINI] Audio encoded successfully')
        except Exception as e:
            print(f'[GEMINI] Error encoding audio: {e}')
            return
        
        # Transcribe and analyze audio content
        analysis_prompt = """请详细分析这段音频中的语音和声音，并以JSON格式返回结果。

请返回以下结构的JSON：
{
    "speech_content": "完整的语音转录文本",
    "emotion_state": "主要情绪状态（快乐、悲伤、愤怒、焦虑、兴奋、平静等）",
    "emotion_intensity": "情绪强度（1-10分）",
    "volume_level": "音量水平（低/中/高）",
    "voice_quality": "声音清晰度（清晰/一般/模糊）",
    "language_feature": "语言和口音特征",
    "speech_speed": "语速（慢/正常/快）",
    "tone_analysis": "语调变化和情感色彩",
    "background_noise": "背景噪音描述",
    "audio_quality": "音频质量评估",
    "psychological_state": "说话者心理状态推断",
    "emotional_stability": "情绪稳定性（稳定/一般/不稳定）",
    "emotion_triggers": "可能的情绪触发因素"
}

请确保返回的是有效的JSON格式，不要包含任何其他文本或说明。"""
        
        response = gemini_client.chat.completions.create(
            model="gemini-3-pro-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in audio analysis. Always return valid JSON format."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analysis_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": audio_url
                            },
                            "mime_type": "audio/webm",
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=4096
        )
        
        analysis_text = response.choices[0].message.content
        print(f'[GEMINI] Audio analysis completed successfully, {analysis_text}')
        
        # Parse JSON response
        try:
            # Try to extract JSON from the response
            import json
            import re
            
            # First, try to extract JSON from markdown code blocks
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, analysis_text, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                # Extract JSON from markdown code block
                json_str = json_match.group(1)
                print(f'[GEMINI] Found JSON in markdown code block')
            else:
                # Try to find JSON object in the response (without markdown)
                json_pattern = r'\{.*\}'
                json_match = re.search(json_pattern, analysis_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    print(f'[GEMINI] Found JSON object in response')
                else:
                    # Fallback: try to parse the entire response as JSON
                    json_str = analysis_text.strip()
                    print(f'[GEMINI] Parsing entire response as JSON')
            
            # Clean up the JSON string
            json_str = json_str.strip()
            
            # Parse the JSON
            audio_analysis = json.loads(json_str)
            
            # Extract values from JSON with defaults
            speech_content = audio_analysis.get('speech_content', '无语音')
            emotion_state = audio_analysis.get('emotion_state', '未检测到')
            emotion_intensity = audio_analysis.get('emotion_intensity', '未知')
            volume_level = audio_analysis.get('volume_level', '未知')
            voice_quality = audio_analysis.get('voice_quality', '正常')
            
            # Limit transcript length for display
            if len(speech_content) > 50:
                speech_content = speech_content[:50] + "..."
                
            print(f'[GEMINI] Successfully parsed audio analysis JSON: {emotion_state}, {volume_level}')
            
        except Exception as e:
            print(f'[GEMINI] Error parsing audio analysis JSON: {e}')
            print(f'[GEMINI] Raw response: {analysis_text}')
            
            # Fallback values if JSON parsing fails
            speech_content = "解析失败"
            emotion_state = "未检测到"
            emotion_intensity = "未知"
            volume_level = "未知"
            voice_quality = "解析错误"
        
        # Generate comprehensive audio detection text
        audio_text = f"语音内容: {speech_content}\n情绪状态: {emotion_state}\n音量: {volume_level}\n音质: {voice_quality}\n分析时间: {datetime.now().strftime('%H:%M:%S')}"
        broadcast_to_clients('audio_detect', audio_text)
        
        # Store analysis results for summary generation
        audio_result = {
            'timestamp': datetime.now().isoformat(),
            'speech_content': speech_content,
            'emotion_state': emotion_state,
            'emotion_intensity': emotion_intensity,
            'volume_level': volume_level,
            'voice_quality': voice_quality,
            'full_analysis': analysis_text,
            'file_path': audio_path
        }
        
        # Add to storage with size limit
        analysis_results['audio_analysis'].append(audio_result)
        if len(analysis_results['audio_analysis']) > MAX_STORED_RESULTS:
            analysis_results['audio_analysis'].pop(0)
        
        # Store emotion history from audio
        emotion_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': 'audio',
            'emotion': emotion_state,
            'intensity': emotion_intensity,
            'confidence': 'high'
        }
        analysis_results['emotion_history'].append(emotion_entry)
        if len(analysis_results['emotion_history']) > MAX_STORED_RESULTS:
            analysis_results['emotion_history'].pop(0)
        
        # Store activity history from audio
        activity_entry = {
            'timestamp': datetime.now().isoformat(),
            'source': 'audio',
            'activity': f"语音活动: {speech_content[:20]}..." if len(speech_content) > 20 else f"语音活动: {speech_content}",
            'intensity': volume_level
        }
        analysis_results['activity_history'].append(activity_entry)
        if len(analysis_results['activity_history']) > MAX_STORED_RESULTS:
            analysis_results['activity_history'].pop(0)
        
        print(f'[STORAGE] Stored audio analysis results. Total stored: {len(analysis_results["audio_analysis"])}')
        
        # Clean up uploaded file
        try:
            gemini_client.files.delete(uploaded_file.name)
            print(f'[GEMINI] Cleaned up audio file: {uploaded_file.name}')
        except:
            pass
        
        print(f'[GEMINI] Audio analysis completed for: {audio_path}')
        
    except Exception as e:
        print(f'[GEMINI] Error analyzing audio: {e}')
        # Fallback to mock data
        html_content = generate_audio_detection_html()
        broadcast_to_clients('audio_detect', html_content)

def run_gemini_video_analysis(video_path):
    """Run video analysis in async context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(analyze_video_with_gemini(video_path))
    finally:
        loop.close()

def run_gemini_audio_analysis(audio_path):
    """Run audio analysis in async context"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(analyze_audio_with_gemini(audio_path))
    finally:
        loop.close()


# Start background timers
def start_background_timers():
    """Start all background timer threads"""
    # Sensor values timer (1 second)
    sensor_thread = threading.Thread(target=sensor_update_timer, daemon=True)
    sensor_thread.start()
    print('[STREAM] Started sensor update timer (1s)')

    # Emotion update timer (3 seconds)
    emotion_thread = threading.Thread(target=emotion_update_timer, daemon=True)
    emotion_thread.start()
    print('[STREAM] Started emotion update timer (3s)')

    # Overall status timer (20 seconds)
    status_thread = threading.Thread(target=overall_status_timer, daemon=True)
    status_thread.start()
    print('[STREAM] Started overall status timer (20s)')

    # Comprehensive summary timer (5 minutes)
    summary_thread = threading.Thread(target=comprehensive_summary_timer, daemon=True)
    summary_thread.start()
    print('[SUMMARY] Started comprehensive summary timer (5m)')


if __name__ == '__main__':
    print("Starting YOLO food detection server with streaming...")
    
    # Start background timers
    start_background_timers()
    
    # Run the app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
