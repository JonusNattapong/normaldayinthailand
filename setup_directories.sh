#!/bin/bash

# Create main directories
mkdir -p models/pretrained
mkdir -p models/finetuned
mkdir -p datasets/raw
mkdir -p datasets/processed
mkdir -p configs/training
mkdir -p configs/model
mkdir -p logs/training
mkdir -p logs/evaluation
mkdir -p checkpoints

# Add .gitkeep to keep empty directories in git
touch models/pretrained/.gitkeep
touch models/finetuned/.gitkeep
touch datasets/raw/.gitkeep
touch datasets/processed/.gitkeep
touch configs/training/.gitkeep
touch configs/model/.gitkeep
touch logs/training/.gitkeep
touch logs/evaluation/.gitkeep
touch checkpoints/.gitkeep

echo "Directory structure created successfully!"