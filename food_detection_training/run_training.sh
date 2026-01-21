#!/bin/bash
# Script to run food detection model training

echo "Setting up environment and starting food detection model training..."

# Activate virtual environment if it exists
if [ -d "../.venv" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
else
    echo "Virtual environment not found at ../.venv"
    exit 1
fi

# Install requirements if not already installed
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    python -m pip install -r requirements.txt
else
    echo "requirements.txt not found"
    exit 1
fi

# Run the training script
echo "Starting training..."
python train_food_model.py

echo "Training script completed."
