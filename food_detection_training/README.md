# Custom Food Detection Model Training

This project provides a complete setup for training a custom YOLOv8n segmentation model specifically optimized for food detection. The system is designed to work with popular food image datasets and is optimized to run on CPU with PyTorch.

## Features
- Trains a YOLOv8n segmentation model for food detection
- Optimized for CPU training with PyTorch
- Primary dataset: VireoFood172 (172 food categories)
- Alternative datasets supported: UNIMIB2016, Food-101, Recipe1M
- Automatic dataset download and preprocessing
- Configurable training parameters (epochs, image size, batch size)

## Dataset Options
1. **VireoFood172** (Primary): Large-scale food dataset with 172 food categories (~140K images)
2. **UNIMIB2016**: Food Segmentation and Classification Dataset with 11,405 food images belonging to 78 classes
3. **Food-101**: Contains 101 food categories with ~1000 images each
4. **Recipe1M**: Dataset with over 1 million cooking recipes and corresponding images

## Setup Instructions
1. Navigate to the food_detection_training directory
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download the VireoFood172 dataset manually (see DOWNLOAD_INSTRUCTIONS.md)
4. Run the training script: `bash run_training.sh`

## Training Configuration
- Model: YOLOv8n-seg (segmentation)
- Device: CPU
- Epochs: 50 (adjustable)
- Image Size: 640x640
- Batch Size: 8 (adjustable for CPU optimization)
- Optimizer: AdamW (for better convergence)

## Usage
To start training the model:
```bash
bash run_training.sh
```

Or run directly:
```bash
python train_food_model.py
```
