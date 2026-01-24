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

# Connected clients
connected_clients = set()

# Load the trained food detection model
# First try to load the trained model from the training directory
#model_path = "food_detection_training/runs/detect/food_det_model/weights/best.pt"
model_path = "yolov8n-seg.pt"
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
        if not filename.lower().endswith('.webm'):
            filename += '.webm'
        
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

def generate_sensor_data():
    """Generate fake sensor data like from a sport watch"""
    return {
        'energy': random.randint(20, 100),
        'sleep': f"{random.uniform(6, 9):.2f} Hours",
        'sport': f"{random.randint(1000, 15000)} Steps",
        'blood': f"{random.randint(70, 120)}",
        'heartRate': f"{random.randint(60, 100)}",
        'calories': f"{random.randint(200, 800)}"
    }

def generate_sensor_html():
    """Generate HTML for sensor values"""
    data = generate_sensor_data()
    return f"""
    <div class="text-green-500">
        <div>Energy: {data['energy']}</div>
        <div>Sleep: {data['sleep']}</div>
        <div>Sport: {data['sport']}</div>
        <div>Blood Pressure: {data['blood']}</div>
        <div>HeartRate: {data['heartRate']}</div>
        <div>Calories: {data['calories']}</div>
    </div>
    """

def generate_emotion_html():
    """Generate HTML for user emotions"""
    emotions = ['Happy', 'Focused', 'Neutral', 'Relaxed', 'Concentrated', 'Thoughtful']
    emotion = random.choice(emotions)
    confidence = random.uniform(0.7, 0.95)
    return f"""
    <div class="text-green-500">
        <div>Emotion: {emotion}</div>
        <div>Confidence: {confidence:.2f}</div>
        <div>Detected: {datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """

def generate_audio_detection_html():
    """Generate HTML for audio detection results"""
    activities = ['Speaking', 'Silent', 'Background noise', 'Conversation', 'Typing']
    activity = random.choice(activities)
    volume = random.randint(20, 80)
    return f"""
    <div class="text-green-500">
        <div>Activity: {activity}</div>
        <div>Volume: {volume} dB</div>
        <div>Duration: {random.randint(1, 10)}s</div>
    </div>
    """

def generate_video_detection_html():
    """Generate HTML for video detection results"""
    activities = ['Working at computer', 'Looking at screen', 'Typing', 'Reading', 'Writing notes']
    activity = random.choice(activities)
    objects = ['person', 'computer', 'keyboard', 'mouse', 'cup']
    detected_objects = random.sample(objects, random.randint(1, 3))
    return f"""
    <div class="text-green-500">
        <div>Activity: {activity}</div>
        <div>Objects: {', '.join(detected_objects)}</div>
        <div>Movement: {random.choice(['Low', 'Medium', 'High'])}</div>
    </div>
    """

def generate_overall_status_html():
    """Generate HTML for overall user status"""
    statuses = [
        'User is actively working and focused',
        'User appears to be in a productive state',
        'User is engaged in computer-based activities',
        'User shows normal work patterns',
        'User is maintaining consistent activity levels'
    ]
    status = random.choice(statuses)
    return f"""
    <div class="text-green-500">
        <div><strong>Overall Status:</strong></div>
        <div>{status}</div>
        <div class="text-xs mt-1">Updated: {datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """

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
            html_content = generate_overall_status_html()
            broadcast_to_clients('overall_status', html_content)
            time.sleep(20)
        except Exception as e:
            print(f'[STREAM] Error in overall status timer: {e}')
            time.sleep(20)

async def analyze_video_with_gemini(video_path):
    """Analyze video using Gemini model"""
    try:
        # This is a placeholder for Gemini integration
        # In real implementation, you would use Google's Gemini API
        print(f'[GEMINI] Analyzing video: {video_path}')
        
        # Simulate API call delay
        await asyncio.sleep(2)
        
        # Generate mock analysis results
        html_content = generate_video_detection_html()
        broadcast_to_clients('video_detect', html_content)
        
        # Generate emotion analysis
        emotion_html = generate_emotion_html()
        broadcast_to_clients('emotion_update', emotion_html)
        
        print(f'[GEMINI] Video analysis completed for: {video_path}')
        
    except Exception as e:
        print(f'[GEMINI] Error analyzing video: {e}')

async def analyze_audio_with_gemini(audio_path):
    """Analyze audio using Gemini model"""
    try:
        # This is a placeholder for Gemini integration
        # In real implementation, you would use Google's Gemini API
        print(f'[GEMINI] Analyzing audio: {audio_path}')
        
        # Simulate API call delay
        await asyncio.sleep(1.5)
        
        # Generate mock analysis results
        html_content = generate_audio_detection_html()
        broadcast_to_clients('audio_detect', html_content)
        
        print(f'[GEMINI] Audio analysis completed for: {audio_path}')
        
    except Exception as e:
        print(f'[GEMINI] Error analyzing audio: {e}')

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


if __name__ == '__main__':
    print("Starting YOLO food detection server with streaming...")
    
    # Start background timers
    start_background_timers()
    
    # Run the app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
