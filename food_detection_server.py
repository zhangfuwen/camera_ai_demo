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

# Load YOLO segmentation model
print("Loading YOLO segmentation model...")
try:
    model = YOLO('yolov8n-seg.pt')  # You can change this to yolov8s-seg.pt, yolov8m-seg.pt, etc. for different sizes
except:
    # If the model isn't downloaded, try to download it first
    print("Downloading YOLO segmentation model...")
    model = YOLO('yolov8n-seg.pt')

# COCO dataset class names (index corresponds to class id)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Food-related categories (indices correspond to COCO dataset class ids)
FOOD_CATEGORIES = {
    44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 
    53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli',
    57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
    72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
    77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster',
    81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
    86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'
}

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
    Detect food items in an image sent via POST request using YOLO segmentation
    Expected format: JSON with 'image' field containing base64 encoded image
    """
    # print("detect_food")
    # app.logger.info("detect_food")
    try:
        print("request", request)
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            print("No image provided")
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL image to numpy array for YOLO
        img_array = np.array(image)
        
        # Perform segmentation with YOLO
        results = model(img_array)
        # print("results", results)
        
        # Process results
        food_results = []
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segmentation masks
            print("boxes", len(boxes))
            print("masks", len(masks))

            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])  # Class id
                    conf = float(box.conf[0])  # Confidence score

                    # Check if this category is food-related
                    if cls in FOOD_CATEGORIES:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Create result entry
                        result_entry = {
                            'label': FOOD_CATEGORIES[cls],
                            'score': conf,
                            'box': {
                                'xmin': float(x1),
                                'ymin': float(y1),
                                'xmax': float(x2),
                                'ymax': float(y2)
                            }
                        }

                        # Add mask if available
                        if masks is not None and i < len(masks):
                            mask = masks[i].data.cpu().numpy()[0]  # Extract mask data
                            result_entry['mask'] = mask.tolist()  # Convert to list for JSON serialization

                        food_results.append(result_entry)

        return jsonify({
            'success': True,
            'detections': food_results,
            'total_food_items': len(food_results)
        })

    except Exception as e:
        print("Exception", e)
        return jsonify({'error': str(e)}), 500

@app.route('/detect_food_stream', methods=['POST'])
def detect_food_stream():
    """
    Detect food items in video stream frames using YOLO segmentation
    Expects multipart/form-data with 'frame' containing image file
    """
    try:
        print("request", request)
        if 'frame' not in request.files:
            return jsonify({'error': 'No frame provided'}), 400

        file = request.files['frame']
        image = Image.open(file.stream)

        # Convert PIL image to numpy array for YOLO
        img_array = np.array(image)

        # Perform segmentation with YOLO
        results = model(img_array)

        # Process results
        food_results = []
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segmentation masks

            if boxes is not None:
                for i, box in enumerate(boxes):
                    cls = int(box.cls[0])  # Class id
                    conf = float(box.conf[0])  # Confidence score

                    # Check if this category is food-related
                    if cls in FOOD_CATEGORIES:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                        # Create result entry
                        result_entry = {
                            'label': FOOD_CATEGORIES[cls],
                            'score': conf,
                            'box': {
                                'xmin': float(x1),
                                'ymin': float(y1),
                                'xmax': float(x2),
                                'ymax': float(y2)
                            }
                        }

                        # Add mask if available
                        if masks is not None and i < len(masks):
                            mask = masks[i].data.cpu().numpy()[0]  # Extract mask data
                            result_entry['mask'] = mask.tolist()  # Convert to list for JSON serialization

                        food_results.append(result_entry)

        return jsonify({
            'success': True,
            'detections': food_results,
            'total_food_items': len(food_results)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    print("Starting YOLO food detection server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
