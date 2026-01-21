# Food Detection Model Training System - Summary

## Directory Structure
- `train_food_model.py`: Main training script that handles dataset download, preprocessing, and model training
- `requirements.txt`: Dependencies needed for training
- `README.md`: Basic instructions for setup and usage
- `run_training.sh`: Executable script to run the training process
- `SUMMARY.md`: This file providing detailed overview of the system

## Features
- Trains a YOLOv8n segmentation model specifically for food detection
- Primary support for VireoFood172 dataset (172 food categories)
- Alternative support for multiple food datasets (UNIMIB2016, Food-101, Recipe1M)
- Optimized for CPU training with PyTorch
- Automatic dataset download and preprocessing for supported datasets
- Configurable training parameters

## Setup Instructions
1. Navigate to the food_detection_training directory
2. Install dependencies: `pip install -r requirements.txt`
3. Download the VireoFood172 dataset manually (follow instructions in DOWNLOAD_INSTRUCTIONS.md)
4. Run the training script: `bash run_training.sh`

## Training Configuration
- Model: YOLOv8n-seg (segmentation)
- Device: CPU (PyTorch)
- Epochs: 50 (can be adjusted)
- Image Size: 640x640
- Batch Size: 8 (optimized for CPU)
- Optimizer: AdamW

## Dataset Information
The system primarily works with the VireoFood172 dataset:
1. **VireoFood172** (Primary): 172 food categories with ~140K images total - requires manual download
2. **UNIMIB2016** (Alternative): 78 food categories with 11,405 images total - automatic download
3. **Food-101** (Alternative): 101 food categories with ~1000 images each - automatic download
4. **Recipe1M** (Alternative): Over 1 million cooking recipes and images - automatic download

The system defaults to VireoFood172 but will fall back to UNIMIB2016 if the primary dataset is not available.

## Expected Output
After training, the model will be saved in `runs/segment/food_seg_model/weights/` with:
- `best.pt`: Best performing model weights
- `last.pt`: Final epoch model weights
- Training/validation metrics and plots

## Customization Options
- Adjust number of epochs by modifying the `epochs` parameter in the train_yolo_model function
- Change image size by modifying the `imgsz` parameter
- Modify batch size with the `batch` parameter
- Switch between different optimizers (SGD, Adam, AdamW)

## Notes
- The training process may take several hours depending on the dataset size and hardware
- CPU training is slower than GPU but more accessible
- The system includes fallback mechanisms if primary dataset is not available
- All models are saved with timestamped directories for easy tracking of experiments
- VireoFood172 dataset requires manual download due to licensing restrictions
