#!/bin/bash

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate ml4hc

# Start TensorBoard in the background and save its process ID (PID)
tensorboard --logdir=logs --port=6006 &
TENSORBOARD_PID=$!

xdg-open http://localhost:6006


# Run the training script
python train_transformer.py

# Kill the TensorBoard process after the training has completed
kill $TENSORBOARD_PID

# Wait 5 seconds for TensorBoard to release the port
sleep 5
