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
import inspect
import logging
import sys

# Configure logging system
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'VERBOSE': '\033[36m',      # Cyan
        'DEBUG': '\033[35m',        # Magenta
        'INFO': '\033[32m',         # Green
        'WARNING': '\033[33m',      # Yellow
        'ERROR': '\033[31m',        # Red
        'CRITICAL': '\033[41m',     # Red background
        'RESET': '\033[0m'          # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Add line number from extra field if available
        line_no = getattr(record, 'line_no', '?')
        
        return f"{log_color}[{record.levelname:8}][LINE {line_no}] {record.getMessage()}{reset}"

# Add VERBOSE level to logging
logging.VERBOSE = 5  # Lower than DEBUG (10)
logging.addLevelName(logging.VERBOSE, 'VERBOSE')

def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.VERBOSE):
        self._log(logging.VERBOSE, message, args, **kwargs)

logging.Logger.verbose = verbose

# Create logger with line number support
def get_logger_with_line(name='food_detection_server'):
    """Get logger that automatically includes line numbers"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.VERBOSE)
        
        # Create colored formatter
        formatter = ColoredFormatter()
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Set logger level to INFO by default
        logger.setLevel(logging.INFO)
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
    
    return logger

# Create global logger
logger = get_logger_with_line()

def log_with_line(level, message):
    """Log message with automatic line number detection"""
    # Go back 2 frames to get the original caller
    frame = inspect.currentframe().f_back.f_back
    line_number = frame.f_lineno
    
    # Create a LogRecord with line number
    record = logger.makeRecord(
        logger.name, level, '', 0, message, (), None
    )
    record.line_no = line_number
    
    logger.handle(record)

# Convenience methods
def log_verbose(message):
    log_with_line(logging.VERBOSE, message)

def log_debug(message):
    log_with_line(logging.DEBUG, message)

def log_info(message):
    log_with_line(logging.INFO, message)

def log_warning(message):
    log_with_line(logging.WARNING, message)

def log_error(message):
    log_with_line(logging.ERROR, message)

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
max_clients = 10  # Limit maximum concurrent connections

# Gemini client configuration (using OpenAI library)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'sk-EplRjGkWQ9CwXK5w10936448E56a46BcB487EeE809C6Bd40')
gemini_client = None

try:
    gemini_client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://api.apiyi.com/v1"
    )
    log_info('[GEMINI] Client initialized successfully with OpenAI library')
except Exception as e:
    log_error(f'[GEMINI] Failed to initialize client: {e}')
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

log_info("Loading food detection model...")
try:
    model = YOLO(model_path)
    log_info(f"Loaded trained model from {model_path}")
except Exception as e:
    log_warning(f"Could not load trained model: {e}")
    log_info(f"Loading default YOLOv8 model instead...")
    model = YOLO(fallback_model_path)

# Get the model's class names
try:
    class_names = model.names  # This gets the class names from the trained model
    log_info(f"Model loaded with {len(class_names)} classes")
except:
    class_names = {}  # Fallback if class names aren't available
    log_warning("Could not retrieve class names from model")

# Create upload directory for videos if it doesn't exist
UPLOAD_FOLDER = 'recorded_videos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    log_info(f"Created upload directory: {UPLOAD_FOLDER}")

# Create upload directory for audio if it doesn't exist
AUDIO_UPLOAD_FOLDER = 'recorded_audio'
if not os.path.exists(AUDIO_UPLOAD_FOLDER):
    os.makedirs(AUDIO_UPLOAD_FOLDER)
    log_info(f"Created audio upload directory: {AUDIO_UPLOAD_FOLDER}")

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

        log_debug("Image shape:" + str(img.shape))
        
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
        log_verbose(f"[VIDEO UPLOAD] Request received at {datetime.now().isoformat()}")
        log_debug(f"[VIDEO UPLOAD] Request headers: {dict(request.headers)}")
        log_debug(f"[VIDEO UPLOAD] Request form data keys: {list(request.form.keys())}")
        log_debug(f"[VIDEO UPLOAD] Request files keys: {list(request.files.keys())}")
        
        # Check if video file is in the request
        if 'video' not in request.files:
            log_error(f"[VIDEO UPLOAD] ERROR: No video file in request")
            return jsonify({"success": False, "error": "No video file provided"}), 400
        
        video_file = request.files['video']
        log_info(f"[VIDEO UPLOAD] Video file received: {video_file.filename}")
        log_debug(f"[VIDEO UPLOAD] Video file content type: {video_file.content_type}")
        
        if video_file.filename == '':
            log_error(f"[VIDEO UPLOAD] ERROR: Empty filename")
            return jsonify({"success": False, "error": "No video file selected"}), 400
        
        # Get timestamp from form data or generate one
        timestamp = request.form.get('timestamp', datetime.now().isoformat())
        log_debug(f"[VIDEO UPLOAD] Client timestamp: {timestamp}")
        
        # Secure the filename and ensure it has the correct extension
        filename = werkzeug.utils.secure_filename(video_file.filename)
        # if not filename.lower().endswith('.webm'):
        #     filename += '.webm'
        
        # Add timestamp prefix to ensure unique ordering
        timestamp_prefix = timestamp.replace(':', '-').replace('.', '-')
        saved_filename = f"{timestamp_prefix}_{filename}"
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        log_debug(f"[VIDEO UPLOAD] Saving file to: {file_path}")
        
        video_file.save(file_path)
        
        # Get file size for logging
        file_size = os.path.getsize(file_path)
        log_info(f"[VIDEO UPLOAD] File saved successfully: {saved_filename}")
        log_info(f"[VIDEO UPLOAD] File size: {file_size / 1024 / 1024:.2f} MB")
        log_debug(f"[VIDEO UPLOAD] Full path: {os.path.abspath(file_path)}")
        
        # List files in upload directory for verification
        try:
            existing_files = os.listdir(app.config['UPLOAD_FOLDER'])
            log_debug(f"[VIDEO UPLOAD] Total files in upload directory: {len(existing_files)}")
            if len(existing_files) <= 5:  # Only list if not too many files
                log_verbose(f"[VIDEO UPLOAD] Files: {sorted(existing_files)}")
        except Exception as e:
            log_warning(f"[VIDEO UPLOAD] Warning: Could not list directory contents: {e}")
        
        log_info(f"[VIDEO UPLOAD] Upload completed successfully at {datetime.now().isoformat()}")
        
        # Trigger Gemini video analysis in background
        gemini_thread = threading.Thread(target=run_gemini_video_analysis, args=(file_path,), daemon=True)
        gemini_thread.start()
        log_info(f"[GEMINI] Started video analysis for: {saved_filename}")
        
        return jsonify({
            "success": True,
            "filename": saved_filename,
            "timestamp": timestamp,
            "path": file_path,
            "size": file_size
        })
    
    except Exception as e:
        log_error(f"[VIDEO UPLOAD] ERROR: {str(e)}")
        log_error(f"[VIDEO UPLOAD] Error type: {type(e).__name__}")
        import traceback
        log_debug(f"[VIDEO UPLOAD] Traceback: {traceback.format_exc()}")
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
        log_verbose(f"[AUDIO UPLOAD] Request received at {datetime.now().isoformat()}")
        log_debug(f"[AUDIO UPLOAD] Request headers: {dict(request.headers)}")
        log_debug(f"[AUDIO UPLOAD] Request form data keys: {list(request.form.keys())}")
        log_debug(f"[AUDIO UPLOAD] Request files keys: {list(request.files.keys())}")
        
        # Check if audio file is in the request
        if 'audio' not in request.files:
            log_error(f"[AUDIO UPLOAD] ERROR: No audio file in request")
            return jsonify({"success": False, "error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        log_debug(f"[AUDIO UPLOAD] Audio file received: {audio_file.filename}")
        log_debug(f"[AUDIO UPLOAD] Audio file content type: {audio_file.content_type}")
        
        if audio_file.filename == '':
            log_error(f"[AUDIO UPLOAD] ERROR: Empty filename")
            return jsonify({"success": False, "error": "No audio file selected"}), 400
        
        # Get timestamp from form data or generate one
        timestamp = request.form.get('timestamp', datetime.now().isoformat())
        log_debug(f"[AUDIO UPLOAD] Client timestamp: {timestamp}")
        
        # Secure the filename and ensure it has the correct extension
        filename = werkzeug.utils.secure_filename(audio_file.filename)
        if not filename.lower().endswith('.webm'):
            filename += '.webm'
        
        # Add timestamp prefix to ensure unique ordering
        timestamp_prefix = timestamp.replace(':', '-').replace('.', '-')
        saved_filename = f"{timestamp_prefix}_{filename}"
        
        # Save the file
        file_path = os.path.join(app.config['AUDIO_UPLOAD_FOLDER'], saved_filename)
        log_debug(f"[AUDIO UPLOAD] Saving file to: {file_path}")
        
        audio_file.save(file_path)
        
        # Get file size for logging
        file_size = os.path.getsize(file_path)
        log_debug(f"[AUDIO UPLOAD] File saved successfully: {saved_filename}")
        log_debug(f"[AUDIO UPLOAD] File size: {file_size / 1024:.2f} KB")
        log_debug(f"[AUDIO UPLOAD] Full path: {os.path.abspath(file_path)}")
        
        # List files in upload directory for verification
        try:
            existing_files = os.listdir(app.config['AUDIO_UPLOAD_FOLDER'])
            log_debug(f"[AUDIO UPLOAD] Total files in audio upload directory: {len(existing_files)}")
            if len(existing_files) <= 5:  # Only list if not too many files
                log_verbose(f"[AUDIO UPLOAD] Files: {sorted(existing_files)}")
        except Exception as e:
            log_warning(f"[AUDIO UPLOAD] Warning: Could not list directory contents: {e}")
        
        log_debug(f"[AUDIO UPLOAD] Upload completed successfully at {datetime.now().isoformat()}")
        
        # Trigger Gemini audio analysis in background
        gemini_thread = threading.Thread(target=run_gemini_audio_analysis, args=(file_path,), daemon=True)
        gemini_thread.start()
        log_debug(f"[GEMINI] Started audio analysis for: {saved_filename}")
        
        return jsonify({
            "success": True,
            "filename": saved_filename,
            "timestamp": timestamp,
            "path": file_path,
            "size": file_size
        })
    
    except Exception as e:
        log_error(f"[AUDIO UPLOAD] ERROR: {str(e)}")
        log_error(f"[AUDIO UPLOAD] Error type: {type(e).__name__}")
        import traceback
        log_debug(f"[AUDIO UPLOAD] Traceback: {traceback.format_exc()}")
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
    client_id = request.sid
    
    # Log connection details for debugging
    log_info(f'[STREAM] New connection attempt - Client: {client_id}, Current: {len(connected_clients)}/{max_clients}')
    
    # Check if we've reached the maximum number of clients
    if len(connected_clients) >= max_clients:
        log_warning(f'[STREAM] Connection rejected - max clients ({max_clients}) reached. Client: {client_id}')
        emit('error', {'message': f'Server is at maximum capacity ({max_clients} clients)'})
        return False  # This will disconnect the client
    
    # Check if this client is already connected (duplicate connection)
    if client_id in connected_clients:
        log_warning(f'[STREAM] Duplicate connection detected: {client_id}')
        return False
    
    log_info(f'[STREAM] Client connected: {client_id} (Total: {len(connected_clients) + 1})')
    connected_clients.add(client_id)
    
    # Send current status to new client
    emit('status', {'message': 'Connected to stream', 'client_count': len(connected_clients)})
    
    # Broadcast updated client count to all clients
    broadcast_to_clients('client_update', {'count': len(connected_clients)})

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    if client_id in connected_clients:
        connected_clients.discard(client_id)
        log_info(f'[STREAM] Client disconnected: {client_id} (Total: {len(connected_clients)})')
        
        # Broadcast updated client count to all clients
        broadcast_to_clients('client_update', {'count': len(connected_clients)})
    else:
        log_debug(f'[STREAM] Unknown client disconnected: {client_id}')

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
        log_debug(f'[STREAM] Broadcasted {message_type} to {len(connected_clients)} clients')
        if message_type == 'audio_detect':
            log_info(f'[STREAM] Broadcasted {message_type} to {len(connected_clients)} clients')

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
    """Generate HTML for sensor values with proper formatting using loop"""
    data = generate_sensor_data()
    
    # Define sensor items with their labels and icons
    sensor_items = [
        {
            'label': '能量值',
            'value': data['energy'],
            'icon': 'M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z'
        },
        {
            'label': '睡眠时长',
            'value': f"{data['sleep']} 小时",
            'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z'
        },
        {
            'label': '运动步数',
            'value': f"{data['sport']} 步",
            'icon': 'M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z'
        },
        {
            'label': '血压',
            'value': data['blood'],
            'icon': 'M9 2a1 1 0 000 2h2a1 1 0 100-2H9z M4 5a2 2 0 012-2 1 1 0 000 2H6a2 2 0 100 4h2a2 2 0 100 4h2a1 1 0 100 2 2 2 0 01-2 2H6a2 2 0 01-2-2V5z'
        },
        {
            'label': '心率',
            'value': f"{data['heartRate']} 次/分",
            'icon': 'M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z'
        },
        {
            'label': '卡路里',
            'value': f"{data['calories']} 卡",
            'icon': 'M9 2a1 1 0 000 2h2a1 1 0 100-2H9z M4 5a2 2 0 012-2 1 1 0 000 2H6a2 2 0 100 4h2a2 2 0 100 4h2a1 1 0 100 2 2 2 0 01-2 2H6a2 2 0 01-2-2V5z'
        }
    ]
    
    # Generate HTML using loop
    html_parts = []
    for item in sensor_items:
        html_parts.append(f"""<div class="flex items-center text-white text-sm">
                                    <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                        <path d="{item['icon']}"/>
                                    </svg>
                                    <span>{item['label']} {item['value']}</span>
                                </div>""")
    
    return ''.join(html_parts)

def generate_emotion_html():
    """Generate HTML for user emotions from stored analysis results using loop"""
    if analysis_results['emotion_history']:
        # Get the most recent emotion analysis
        latest_emotion = analysis_results['emotion_history'][-1]
        
        # Define emotion items with their labels and icons
        emotion_items = [
            {
                'label': '情绪状态',
                'value': latest_emotion['emotion'],
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z'
            },
            {
                'label': '强度',
                'value': latest_emotion['intensity'],
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z'
            },
            {
                'label': '检测时间',
                'value': datetime.fromisoformat(latest_emotion['timestamp']).strftime('%H:%M:%S'),
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z'
            }
        ]
        
        # Generate HTML using loop
        html_parts = []
        for item in emotion_items:
            html_parts.append(f"""<div class="flex items-center text-white text-sm">
                                        <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="{item['icon']}"/>
                                        </svg>
                                        <span>{item['label']} {item['value']}</span>
                                    </div>""")
        
        return ''.join(html_parts)
    else:
        log_debug("[EMOTION] No emotion analysis results available")
        # Return default emotion HTML when no data available
        default_items = [
            {
                'label': '情绪状态',
                'value': '等待分析数据',
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z'
            }
        ]
        
        html_parts = []
        for item in default_items:
            html_parts.append(f"""<div class="flex items-center text-white text-sm">
                                        <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="{item['icon']}"/>
                                        </svg>
                                        <span>{item['label']} {item['value']}</span>
                                    </div>""")
        
        return ''.join(html_parts)

def generate_audio_detection_html():
    """Generate HTML for audio detection results from stored analysis results using loop"""
    if analysis_results['audio_analysis']:
        # Get the most recent audio analysis
        latest_audio = analysis_results['audio_analysis'][-1]
        
        # Define audio items with their labels and icons
        audio_items = [
            {
                'label': '语音内容',
                'value': latest_audio['speech_content'],
                'icon': 'M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z'
            },
            {
                'label': '情绪状态',
                'value': latest_audio['emotion_state'],
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z'
            },
            {
                'label': '音量',
                'value': latest_audio['volume_level'],
                'icon': 'M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.984 5.984 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.983 3.983 0 00-1.172-2.828 1 1 0 010-1.415z'
            },
            {
                'label': '音质',
                'value': latest_audio['voice_quality'],
                'icon': 'M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM12.293 7.293a1 1 0 011.414 0L15 8.586l1.293-1.293a1 1 0 111.414 1.414L16.414 10l1.293 1.293a1 1 0 01-1.414 1.414L15 11.414l-1.293 1.293a1 1 0 01-1.414-1.414L13.586 10l-1.293-1.293a1 1 0 010-1.414z'
            },
            {
                'label': '分析时间',
                'value': datetime.fromisoformat(latest_audio['timestamp']).strftime('%H:%M:%S'),
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z'
            }
        ]
        
        # Generate HTML using loop
        html_parts = []
        for item in audio_items:
            html_parts.append(f"""<div class="flex items-center text-white text-sm">
                                        <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="{item['icon']}"/>
                                        </svg>
                                        <span>{item['label']} {item['value']}</span>
                                    </div>""")
        
        return ''.join(html_parts)
    else:
        log_debug("[AUDIO] No audio analysis results available")
        # Return default audio HTML when no data available
        default_items = [
            {
                'label': '音频分析',
                'value': '等待分析数据',
                'icon': 'M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z'
            }
        ]
        
        html_parts = []
        for item in default_items:
            html_parts.append(f"""<div class="flex items-center text-white text-sm">
                                        <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="{item['icon']}"/>
                                        </svg>
                                        <span>{item['label']} {item['value']}</span>
                                    </div>""")
        
        return ''.join(html_parts)

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
            log_warning(f'[STREAM] Error in sensor timer: {e}')
            time.sleep(1)

def emotion_update_timer():
    """Timer for updating emotions every 3 seconds"""
    while True:
        try:
            html_content = generate_emotion_html()
            broadcast_to_clients('emotion_update', html_content)
            time.sleep(3)
        except Exception as e:
            log_warning(f'[STREAM] Error in emotion timer: {e}')
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
            log_warning(f'[STREAM] Error in overall status timer: {e}')
            time.sleep(20)


async def analyze_video_with_gemini(video_path):
    """Analyze video using Gemini model"""
    try:
        if not gemini_client:
            log_warning('[GEMINI] Client not available, using fallback')
            # Fallback to mock data
            html_content = generate_video_detection_html()
            broadcast_to_clients('video_detect', html_content)
            emotion_html = generate_emotion_html()
            broadcast_to_clients('emotion_update', emotion_html)
            return
        
        log_info(f'[GEMINI] Analyzing video: {video_path}')
        
        # Check if file exists and get file info
        import os
        if not os.path.exists(video_path):
            log_error(f'[GEMINI] Error: Video file does not exist: {video_path}')
            return
        
        file_size = os.path.getsize(video_path)
        log_debug(f'[GEMINI] Video file info - Path: {video_path}, Size: {file_size} bytes')
        
        # Test Gemini client connectivity
        try:
            log_debug(f'[GEMINI] Testing client connectivity...')
            test_response = gemini_client.chat.completions.create(
                model='gemini-3-pro-preview',
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=50
            )
            log_debug(f'[GEMINI] Client connectivity test passed')
        except Exception as e:
            log_error(f'[GEMINI] Client connectivity test failed: {e}')
            return
        
        # Read and encode video file
        try:
            with open(video_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode()
                video_url = f"data:video/mp4;base64,{video_b64}"
            log_debug(f'[GEMINI] Video encoded successfully')
        except Exception as e:
            log_error(f'[GEMINI] Error encoding video: {e}')
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
    "emotion_cause": "可能的情绪原因",
    "eating_drinking": "是否在进食或饮用饮料（是/否）, 如果是，请描述具体的食物或饮料及能量和是否健康"
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
        log_info(f'[GEMINI] Video analysis completed successfully')
        eating_drinking = '未检测到'
        
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
            eating_drinking = video_analysis.get('eating_drinking', '未检测到')
            
            log_info(f'[GEMINI] Successfully parsed video analysis JSON: {emotion_state}, {activity_level}')
            
        except Exception as e:
            log_error(f'[GEMINI] Error parsing video analysis JSON: {e}')
            log_error(f'[GEMINI] Raw response: {analysis_text}')
            
            # Fallback values if JSON parsing fails
            emotion_state = "解析失败"
            emotion_intensity = "未知"
            activity_level = "未知"
            overall_state = "解析错误"
        
        # Generate comprehensive video detection HTML using loop format
        video_items = [
            {
                'label': '进食情况',
                'value': eating_drinking,
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z'
            },
            # {
            #     'label': '心理状态',
            #     'value': overall_state,
            #     'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z'
            # },
            # {
            #     'label': '情绪强度',
            #     'value': emotion_intensity,
            #     'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z'
            # },
            # {
            #     'label': '分析时间',
            #     'value': datetime.now().strftime('%H:%M:%S'),
            #     'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z'
            # }
        ]
        
        # Generate HTML using loop
        html_parts = []
        for item in video_items:
            html_parts.append(f"""<div class="flex items-center text-white text-sm">
                                        <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="{item['icon']}"/>
                                        </svg>
                                        <span>{item['label']} {item['value']}</span>
                                    </div>""")
        
        video_html = ''.join(html_parts)
        broadcast_to_clients('video_detect', video_html)
        
        # Generate detailed emotion HTML using loop format
        emotion_items = [
            {
                'label': '主要情绪',
                'value': emotion_state,
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z'
            },
            {
                'label': '检测时间',
                'value': datetime.now().strftime('%H:%M:%S'),
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z'
            },
            {
                'label': '分析模型',
                'value': 'Gemini-3-Pro',
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z'
            }
        ]
        
        # Generate HTML using loop
        html_parts = []
        for item in emotion_items:
            html_parts.append(f"""<div class="flex items-center text-white text-sm">
                                        <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="{item['icon']}"/>
                                        </svg>
                                        <span>{item['label']} {item['value']}</span>
                                    </div>""")
        
        emotion_html = ''.join(html_parts)
        broadcast_to_clients('emotion_update', emotion_html)
        
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
        log_info(f'[STORAGE] Stored video analysis result: {analysis_results["video_analysis"][-1]}')
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
        
        log_info(f'[STORAGE] Stored video analysis results. Total stored: {len(analysis_results["video_analysis"])}')
        log_info(f'[GEMINI] Video analysis completed for: {video_path}')
        
    except Exception as e:
        log_error(f'[GEMINI] Error analyzing video: {e}')
        # Fallback to mock data
        html_content = generate_video_detection_html()
        broadcast_to_clients('video_detect', html_content)
        emotion_html = generate_emotion_html()
        broadcast_to_clients('emotion_update', emotion_html)

async def generate_comprehensive_summary():
    """Generate comprehensive summary using stored analysis results in JSON format"""
    try:
        if not gemini_client:
            log_info('[SUMMARY] Gemini client not available, using fallback')
            return generate_mock_summary_json()
        
        # Check if we have enough data for meaningful summary
        total_analyses = len(analysis_results['video_analysis']) + len(analysis_results['audio_analysis'])
        if total_analyses < 2:
            log_info('[SUMMARY] Insufficient data for comprehensive summary')
            return generate_mock_summary_json()
        
        log_info(f'[SUMMARY] Generating comprehensive summary from {total_analyses} analyses')
        
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
        summary_prompt = f"""基于以下分析结果，请生成一个综合性的用户状态总结，并以JSON格式返回：

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

请返回以下JSON结构：
{{
    "emotion_trend_analysis": {{
        "main_emotion_pattern": "主要情绪模式",
        "emotion_change_trend": "情绪变化趋势",
        "emotion_stability_assessment": "情绪稳定性评估"
    }},
    "activity_pattern_analysis": {{
        "main_activity_type": "主要活动类型",
        "activity_intensity_change": "活动强度变化",
        "work_rest_balance": "工作/休息平衡"
    }},
    "communication_pattern_analysis": {{
        "voice_activity_frequency": "语音活动频率",
        "communication_emotion_features": "沟通情绪特征",
        "expression_style_characteristics": "表达方式特点"
    }},
    "comprehensive_status_assessment": {{
        "current_overall_status": "当前整体状态",
        "stress_level_assessment": "压力水平评估",
        "efficiency_status_assessment": "效率状态评估"
    }},
    "suggestions_and_reminders": {{
        "health_suggestions": "健康建议（重点关注饮食健康和营养均衡）" if has_eating_activity else "健康建议",
        "work_efficiency_suggestions": "工作效率建议",
        "emotion_management_suggestions": "情绪管理建议"
    }}{', "diet_health_specific_suggestions": {{"eating_time_regularity_suggestions": "进食时间规律性建议", "nutrition_matching_suggestions": "营养搭配建议", "healthy_eating_habit_suggestions": "健康饮食习惯建议", "digestive_health_reminders": "消化健康提醒"}}' if has_eating_activity else ''}
}}

请确保返回有效的JSON格式，每个值只输出一句简短描述。{'如果检测到进食活动，请在健康建议部分特别关注饮食健康和营养均衡。' if has_eating_activity else ''}"""

        # Generate summary using Gemini
        response = gemini_client.chat.completions.create(
            model="gemini-3-pro-preview",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3,
            max_tokens=4096
        )
        
        summary_text = response.choices[0].message.content
        log_info(f'[SUMMARY] Comprehensive summary generated successfully')
        
        # Parse JSON response
        try:
            import json
            import re
            
            # Try to extract JSON from the response
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, summary_text, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                json_str = json_match.group(1)
                log_debug('[SUMMARY] Found JSON in markdown code block')
            else:
                json_pattern = r'\{.*\}'
                json_match = re.search(json_pattern, summary_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    log_debug('[SUMMARY] Found JSON object in response')
                else:
                    json_str = summary_text.strip()
                    log_debug('[SUMMARY] Parsing entire response as JSON')
            
            json_str = json_str.strip()
            summary_data = json.loads(json_str)
            
            # Add metadata
            summary_data['metadata'] = {
                'generated_at': datetime.now().isoformat(),
                'total_analyses': total_analyses,
                'video_analyses_count': len(recent_video),
                'audio_analyses_count': len(recent_audio),
                'has_eating_activity': has_eating_activity,
                'eating_activities_count': len(eating_activities)
            }
            
            log_info(f'[SUMMARY] Successfully parsed summary JSON')
            return summary_data
            
        except Exception as e:
            log_error(f'[SUMMARY] Error parsing summary JSON: {e}')
            log_error(f'[SUMMARY] Raw response: {summary_text}')
            return generate_mock_summary_json()
        
    except Exception as e:
        log_info(f'[SUMMARY] Error generating comprehensive summary: {e}')
        return generate_mock_summary_json()

def generate_mock_summary_json():
    """Generate mock summary JSON when Gemini is unavailable"""
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
    
    # Generate base summary JSON
    mock_summary = {
        "emotion_trend_analysis": {
            "main_emotion_pattern": random.choice(emotion_states),
            "emotion_change_trend": "情绪状态相对稳定",
            "emotion_stability_assessment": "良好"
        },
        "activity_pattern_analysis": {
            "main_activity_type": "电脑工作",
            "activity_intensity_change": random.choice(activity_levels),
            "work_rest_balance": "需要注意适当休息"
        },
        "communication_pattern_analysis": {
            "voice_activity_frequency": "适中",
            "communication_emotion_features": "积极正面",
            "expression_style_characteristics": "清晰流畅"
        },
        "comprehensive_status_assessment": {
            "current_overall_status": "良好",
            "stress_level_assessment": "中等",
            "efficiency_status_assessment": "高效"
        },
        "suggestions_and_reminders": {
            "health_suggestions": "定时休息，保护视力" if not has_eating_activity else "定时休息，注意饮食营养均衡，避免暴饮暴食，建议细嚼慢咽有助于消化健康",
            "work_efficiency_suggestions": "保持专注，适当调整",
            "emotion_management_suggestions": "保持积极心态"
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_analyses": len(analysis_results.get('video_analysis', [])) + len(analysis_results.get('audio_analysis', [])),
            "video_analyses_count": len(analysis_results.get('video_analysis', [])),
            "audio_analyses_count": len(analysis_results.get('audio_analysis', [])),
            "has_eating_activity": has_eating_activity,
            "eating_activities_count": len(eating_activities),
            "is_mock": True
        }
    }
    
    # Add eating-specific advice if eating activity detected
    if has_eating_activity:
        mock_summary["diet_health_specific_suggestions"] = {
            "eating_time_regularity_suggestions": "建议固定用餐时间，避免不规律进食",
            "nutrition_matching_suggestions": "注意荤素搭配，保证营养均衡",
            "healthy_eating_habit_suggestions": "细嚼慢咽，避免边工作边进食",
            "digestive_health_reminders": "饭后适当休息，有助于消化吸收"
        }
    
    return mock_summary

def generate_summary_html(summary_data):
    """Generate HTML for summary display from JSON data"""
    if isinstance(summary_data, str):
        # Handle legacy string input
        return f"""
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <div class="flex items-center mb-4">
                <svg class="w-6 h-6 mr-2 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z M4 5a2 2 0 012-2 1 1 0 000 2H6a2 2 0 100 4h2a2 2 0 100 4h2a1 1 0 100 2 2 2 0 01-2 2H6a2 2 0 01-2-2V5z"/>
                </svg>
                <h3 class="text-lg font-bold text-white">综合状态总结</h3>
            </div>
            <p class="text-gray-400 text-sm mb-4">生成时间: {datetime.now().strftime('%H:%M:%S')}</p>
            <div class="text-gray-300 whitespace-pre-wrap">{summary_data}</div>
        </div>
        """
    
    # Generate HTML from JSON data
    html_sections = []
    
    # Header
    html_sections.append(f"""
    <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 shadow-lg">
        <div class="flex items-center justify-between mb-6">
            <div class="flex items-center">
                <svg class="w-6 h-6 mr-3 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z M4 5a2 2 0 012-2 1 1 0 000 2H6a2 2 0 100 4h2a2 2 0 100 4h2a1 1 0 100 2 2 2 0 01-2 2H6a2 2 0 01-2-2V5z"/>
                </svg>
                <h3 class="text-xl font-bold text-white">综合状态总结</h3>
            </div>
            <span class="text-sm text-gray-400">{datetime.now().strftime('%H:%M:%S')}</span>
        </div>
    """)
    
    # Emotion Trend Analysis
    if 'emotion_trend_analysis' in summary_data:
        emotion = summary_data['emotion_trend_analysis']
        html_sections.append(f"""
        <div class="mb-6 p-4 bg-gray-700 rounded-lg border border-gray-600">
            <div class="flex items-center mb-3">
                <svg class="w-5 h-5 mr-2 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z"/>
                </svg>
                <h4 class="text-lg font-semibold text-white">整体情绪趋势分析</h4>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="flex items-start">
                    <span class="text-blue-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">主要情绪模式</p>
                        <p class="text-white font-medium">{emotion.get('main_emotion_pattern', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-green-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">情绪变化趋势</p>
                        <p class="text-white font-medium">{emotion.get('emotion_change_trend', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-yellow-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">情绪稳定性评估</p>
                        <p class="text-white font-medium">{emotion.get('emotion_stability_assessment', '未知')}</p>
                    </div>
                </div>
            </div>
        </div>
        """)
    
    # Activity Pattern Analysis
    if 'activity_pattern_analysis' in summary_data:
        activity = summary_data['activity_pattern_analysis']
        html_sections.append(f"""
        <div class="mb-6 p-4 bg-gray-700 rounded-lg border border-gray-600">
            <div class="flex items-center mb-3">
                <svg class="w-5 h-5 mr-2 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"/>
                </svg>
                <h4 class="text-lg font-semibold text-white">活动模式分析</h4>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="flex items-start">
                    <span class="text-blue-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">主要活动类型</p>
                        <p class="text-white font-medium">{activity.get('main_activity_type', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-green-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">活动强度变化</p>
                        <p class="text-white font-medium">{activity.get('activity_intensity_change', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-yellow-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">工作/休息平衡</p>
                        <p class="text-white font-medium">{activity.get('work_rest_balance', '未知')}</p>
                    </div>
                </div>
            </div>
        </div>
        """)
    
    # Communication Pattern Analysis
    if 'communication_pattern_analysis' in summary_data:
        communication = summary_data['communication_pattern_analysis']
        html_sections.append(f"""
        <div class="mb-6 p-4 bg-gray-700 rounded-lg border border-gray-600">
            <div class="flex items-center mb-3">
                <svg class="w-5 h-5 mr-2 text-cyan-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z"/>
                </svg>
                <h4 class="text-lg font-semibold text-white">沟通模式分析</h4>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="flex items-start">
                    <span class="text-blue-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">语音活动频率</p>
                        <p class="text-white font-medium">{communication.get('voice_activity_frequency', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-green-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">沟通情绪特征</p>
                        <p class="text-white font-medium">{communication.get('communication_emotion_features', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-yellow-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">表达方式特点</p>
                        <p class="text-white font-medium">{communication.get('expression_style_characteristics', '未知')}</p>
                    </div>
                </div>
            </div>
        </div>
        """)
    
    # Comprehensive Status Assessment
    if 'comprehensive_status_assessment' in summary_data:
        status = summary_data['comprehensive_status_assessment']
        html_sections.append(f"""
        <div class="mb-6 p-4 bg-gray-700 rounded-lg border border-gray-600">
            <div class="flex items-center mb-3">
                <svg class="w-5 h-5 mr-2 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z"/>
                </svg>
                <h4 class="text-lg font-semibold text-white">综合状态评估</h4>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="flex items-start">
                    <span class="text-blue-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">当前整体状态</p>
                        <p class="text-white font-medium">{status.get('current_overall_status', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-green-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">压力水平评估</p>
                        <p class="text-white font-medium">{status.get('stress_level_assessment', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-yellow-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">效率状态评估</p>
                        <p class="text-white font-medium">{status.get('efficiency_status_assessment', '未知')}</p>
                    </div>
                </div>
            </div>
        </div>
        """)
    
    # Suggestions and Reminders
    if 'suggestions_and_reminders' in summary_data:
        suggestions = summary_data['suggestions_and_reminders']
        html_sections.append(f"""
        <div class="mb-6 p-4 bg-gray-700 rounded-lg border border-gray-600">
            <div class="flex items-center mb-3">
                <svg class="w-5 h-5 mr-2 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M11 3a1 1 0 10-2 0v1a1 1 0 102 0V3zM15.657 5.757a1 1 0 00-1.414-1.414l-.707.707a1 1 0 001.414 1.414l.707-.707zM18 10a1 1 0 01-1 1h-1a1 1 0 110-2h1a1 1 0 011 1zM5.05 6.464A1 1 0 106.464 5.05l-.707-.707a1 1 0 00-1.414 1.414l.707.707zM5 10a1 1 0 01-1 1H3a1 1 0 110-2h1a1 1 0 011 1zM8 16v-1a1 1 0 112 0v1a1 1 0 11-2 0zM12 14a1 1 0 00-.707.293l-.707.707a1 1 0 101.414 1.414l.707-.707A1 1 0 0012 14zM5.757 14.243a1 1 0 00-1.414 0l-.707.707a1 1 0 101.414 1.414l.707-.707a1 1 0 000-1.414z"/>
                </svg>
                <h4 class="text-lg font-semibold text-white">建议和提醒</h4>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="flex items-start">
                    <span class="text-blue-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">健康建议</p>
                        <p class="text-white font-medium">{suggestions.get('health_suggestions', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-green-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">工作效率建议</p>
                        <p class="text-white font-medium">{suggestions.get('work_efficiency_suggestions', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-yellow-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">情绪管理建议</p>
                        <p class="text-white font-medium">{suggestions.get('emotion_management_suggestions', '未知')}</p>
                    </div>
                </div>
            </div>
        </div>
        """)
    
    # Diet Health Specific Suggestions (if available)
    if 'diet_health_specific_suggestions' in summary_data:
        diet = summary_data['diet_health_specific_suggestions']
        html_sections.append(f"""
        <div class="mb-6 p-4 bg-green-900 rounded-lg border border-green-700">
            <div class="flex items-center mb-3">
                <svg class="w-5 h-5 mr-2 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M10 2a8 8 0 100 16 8 8 0 000-16zm0 14a6 6 0 110-12 6 6 0 010 12zm0-10a1 1 0 00-1 1v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6a1 1 0 00-1-1z"/>
                </svg>
                <h4 class="text-lg font-semibold text-white">饮食健康专项建议</h4>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div class="flex items-start">
                    <span class="text-green-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">进食时间规律性建议</p>
                        <p class="text-white font-medium">{diet.get('eating_time_regularity_suggestions', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-green-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">营养搭配建议</p>
                        <p class="text-white font-medium">{diet.get('nutrition_matching_suggestions', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-green-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">健康饮食习惯建议</p>
                        <p class="text-white font-medium">{diet.get('healthy_eating_habit_suggestions', '未知')}</p>
                    </div>
                </div>
                <div class="flex items-start">
                    <span class="text-green-400 mr-2">•</span>
                    <div>
                        <p class="text-sm text-gray-300">消化健康提醒</p>
                        <p class="text-white font-medium">{diet.get('digestive_health_reminders', '未知')}</p>
                    </div>
                </div>
            </div>
        </div>
        """)
    
    # Metadata
    if 'metadata' in summary_data:
        metadata = summary_data['metadata']
        data_source_color = 'text-green-400' if not metadata.get('is_mock', False) else 'text-yellow-400'
        data_source_text = 'AI 分析' if not metadata.get('is_mock', False) else '模拟数据'
        
        html_sections.append(f"""
        <div class="p-4 bg-gray-900 rounded-lg border border-gray-700">
            <div class="flex items-center mb-3">
                <svg class="w-5 h-5 mr-2 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z"/>
                </svg>
                <h4 class="text-lg font-semibold text-white">数据统计</h4>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="text-center">
                    <p class="text-2xl font-bold text-blue-400">{metadata.get('total_analyses', 0)}</p>
                    <p class="text-sm text-gray-400">总分析次数</p>
                </div>
                <div class="text-center">
                    <p class="text-2xl font-bold text-green-400">{metadata.get('video_analyses_count', 0)}</p>
                    <p class="text-sm text-gray-400">视频分析次数</p>
                </div>
                <div class="text-center">
                    <p class="text-2xl font-bold text-cyan-400">{metadata.get('audio_analyses_count', 0)}</p>
                    <p class="text-sm text-gray-400">音频分析次数</p>
                </div>
                <div class="text-center">
                    <p class="text-2xl font-bold text-purple-400">{'是' if metadata.get('has_eating_activity', False) else '否'}</p>
                    <p class="text-sm text-gray-400">检测到进食活动</p>
                </div>
            </div>
            {f'<div class="mt-4 text-center"><p class="text-lg font-medium text-green-400">进食活动次数: {metadata.get("eating_activities_count", 0)}</p></div>' if metadata.get('has_eating_activity', False) else ''}
            <div class="mt-4 text-center">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-700 text-gray-300">
                    <svg class="w-4 h-4 mr-2 {data_source_color}" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z M4 5a2 2 0 012-2 1 1 0 000 2H6a2 2 0 100 4h2a2 2 0 100 4h2a1 1 0 100 2 2 2 0 01-2 2H6a2 2 0 01-2-2V5z"/>
                    </svg>
                    数据来源: {data_source_text}
                </span>
            </div>
        </div>
        """)
    
    # Close main container
    html_sections.append("</div>")
    
    return ''.join(html_sections)

async def analyze_audio_with_gemini(audio_path):
    """Analyze audio using Gemini model"""
    try:
        if not gemini_client:
            log_error('[GEMINI] Client not available, using fallback')
            # Fallback to mock data
            html_content = generate_audio_detection_html()
            broadcast_to_clients('audio_detect', html_content)
            return



        # print file info
        import os
        if not os.path.exists(audio_path):
            log_error(f'[GEMINI] Error: Audio file does not exist: {audio_path}')
            return
        file_size = os.path.getsize(audio_path)
        log_info(f'[GEMINI] Analyzing audio: {audio_path}, File Size: {file_size} bytes')

        # Read and encode audio file
        try:
            with open(audio_path, "rb") as f:
                audio_b64 = base64.b64encode(f.read()).decode()
                audio_url = f"data:audio/webm;base64,{audio_b64}"
            log_info(f'[GEMINI] Audio encoded successfully')
        except Exception as e:
            log_error(f'[GEMINI] Error encoding audio: {e}')
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
        log_info(f'[GEMINI] Audio analysis completed successfully, {analysis_text}')
        
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
                
            log_info(f'[GEMINI] Successfully parsed audio analysis JSON: {emotion_state}, {volume_level}')
            
        except Exception as e:
            log_error(f'[GEMINI] Error parsing audio analysis JSON: {e}')
            log_error(f'[GEMINI] Raw response: {analysis_text}')
            
            # Fallback values if JSON parsing fails
            speech_content = "解析失败"
            emotion_state = "未检测到"
            emotion_intensity = "未知"
            volume_level = "未知"
            voice_quality = "解析错误"
        
        # Generate comprehensive audio detection HTML using loop format
        audio_items = [
            {
                'label': '语音内容',
                'value': speech_content,
                'icon': 'M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07z'
            },
            {
                'label': '情绪状态',
                'value': emotion_state,
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z'
            },
            {
                'label': '音量',
                'value': volume_level,
                'icon': 'M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.984 5.984 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.983 3.983 0 00-1.172-2.828 1 1 0 010-1.415z'
            },
            {
                'label': '音质',
                'value': voice_quality,
                'icon': 'M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM12.293 7.293a1 1 0 011.414 0L15 8.586l1.293-1.293a1 1 0 111.414 1.414L16.414 10l1.293 1.293a1 1 0 01-1.414 1.414L15 11.414l-1.293 1.293a1 1 0 01-1.414-1.414L13.586 10l-1.293-1.293a1 1 0 010-1.414z'
            },
            {
                'label': '分析时间',
                'value': datetime.now().strftime('%H:%M:%S'),
                'icon': 'M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z'
            }
        ]
        
        # Generate HTML using loop
        html_parts = []
        for item in audio_items:
            html_parts.append(f"""<div class="flex items-center text-white text-sm">
                                        <svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="{item['icon']}"/>
                                        </svg>
                                        <span>{item['label']} {item['value']}</span>
                                    </div>""")
        
        audio_html = ''.join(html_parts)
        broadcast_to_clients('audio_detect', audio_html)
        
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
        
        log_debug(f'[STORAGE] Stored audio analysis results. Total stored: {len(analysis_results["audio_analysis"])}')
        log_debug(f'[GEMINI] Audio analysis completed for: {audio_path}')
        broadcast_to_clients('audio_detect', audio_text)
        
    except Exception as e:
        log_error(f'[GEMINI] Error analyzing audio: {e}')
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


def cleanup_stale_connections():
    """Clean up stale connections that might be hanging"""
    while True:
        try:
            # Log current connection count every 30 seconds
            log_info(f'[STREAM] Active connections: {len(connected_clients)}')
            
            # If we have too many connections, this might indicate a problem
            if len(connected_clients) > max_clients * 0.8:
                log_warning(f'[STREAM] High connection count detected: {len(connected_clients)}')
            
            time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            log_error(f'[STREAM] Error in connection cleanup: {e}')
            time.sleep(30)

# Start background timers
def start_background_timers():
    """Start all background timer threads"""
    # Connection monitoring thread
    cleanup_thread = threading.Thread(target=cleanup_stale_connections, daemon=True)
    cleanup_thread.start()
    log_info('[STREAM] Started connection monitoring (30s)')
    
    # Sensor values timer (1 second)
    sensor_thread = threading.Thread(target=sensor_update_timer, daemon=True)
    sensor_thread.start()
    log_info('[STREAM] Started sensor update timer (1s)')

    # Emotion update timer (3 seconds)
    emotion_thread = threading.Thread(target=emotion_update_timer, daemon=True)
    emotion_thread.start()
    log_info('[STREAM] Started emotion update timer (3s)')

    # Overall status timer (20 seconds)
    status_thread = threading.Thread(target=overall_status_timer, daemon=True)
    status_thread.start()
    log_info('[STREAM] Started overall status timer (20s)')


if __name__ == '__main__':
    print("Starting YOLO food detection server with streaming...")
    
    # Start background timers
    start_background_timers()
    
    # Run the app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
