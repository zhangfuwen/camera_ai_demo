#!/bin/bash

# Script to start the food detection service
echo "Starting Food Detection Service..."

# Check if required model exists
if [ ! -f "food_detection_training/runs/detect/food_det_model/weights/best.pt" ]; then
    echo "Warning: Trained model not found, will use default YOLOv8 model"
fi

# Start the food detection service
echo "Starting food detection service on port 5000..."
python3 food_detection_server.py

echo "Service stopped."
EOL
