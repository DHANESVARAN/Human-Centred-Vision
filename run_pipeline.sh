#!/bin/bash

# Exit the script immediately if any command fails
set -e

echo "========================================="
echo " Starting CBL-DLT Hand Gesture Pipeline"
echo "========================================="

# 1. Activate the virtual environment
echo "[INFO] Activating virtual environment..."
source cnn_env/bin/activate

# 2. Prepare the dataset
echo "[INFO] Running Step 1: Data Preparation..."
python3 prepare_data.py

# 3. Train the model
echo "[INFO] Running Step 2: Model Training..."
python3 train_model.py

# 4. Evaluate the model on the unseen subject
echo "[INFO] Running Step 3: Evaluation..."
python3 evaluate.py

echo "========================================="
echo " Pipeline completed successfully!"
echo "========================================="
