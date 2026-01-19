"""
Configuration file for the Explainability Project.
Contains paths, hyperparameters, and settings.
"""

import os
from pathlib import Path

# Project paths - go up one level from src/ to project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "CUB_200_2011" / "CUB_200_2011"
IMAGES_DIR = DATA_ROOT / "images"
PARTS_DIR = DATA_ROOT / "parts"
ATTRIBUTES_DIR = DATA_ROOT / "attributes"

# Data files
IMAGES_FILE = DATA_ROOT / "images.txt"
TRAIN_TEST_SPLIT_FILE = DATA_ROOT / "train_test_split.txt"
IMAGE_CLASS_LABELS_FILE = DATA_ROOT / "image_class_labels.txt"
CLASSES_FILE = DATA_ROOT / "classes.txt"
BOUNDING_BOXES_FILE = DATA_ROOT / "bounding_boxes.txt"
PART_LOCS_FILE = PARTS_DIR / "part_locs.txt"
PARTS_FILE = PARTS_DIR / "parts.txt"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, RESULTS_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Image settings
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model training settings
NUM_CLASSES = 200
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Explainability settings
EXPLAINABILITY_METHODS = ['integrated_gradients', 'input_gradients', 'lime', 'kernel_shap']
N_SAMPLES_LIME = 1000
N_SAMPLES_SHAP = 300 #INFO: numero di samples per Kernel SHAP

# Part annotation settings (15 parts in CUB-200)
NUM_PARTS = 15
PART_NAMES = [
    'back', 'beak', 'belly', 'breast', 'crown',
    'forehead', 'left_eye', 'left_leg', 'left_wing', 'nape',
    'right_eye', 'right_leg', 'right_wing', 'tail', 'throat'
]

# Evaluation settings
PART_RADIUS = 10  # Radius around part center for creating ground-truth masks
TOP_K_PERCENT = [5, 10, 15, 20, 25]  # Percentage of top attribution pixels to consider
