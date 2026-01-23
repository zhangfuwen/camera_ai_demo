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
