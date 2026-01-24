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

app = Flask(__name__, static_folder='.', template_folder='.')
CORS(app)  # Enable CORS for all routes

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

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



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


if __name__ == '__main__':
    print("Starting YOLO food detection server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
