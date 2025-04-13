import torch

# Device selection
device = 'cpu'
try:
    if torch.backends.mps.is_available():
        device = 'mps'
except:
    if torch.cuda.is_available():
        device = 'cuda'

# Training configuration
PRE_TRAINED = False
PRE_TRAINED_PATH = 'MODEL_PATH'

BATCH_SIZE = 4
VAL_BATCH_SIZE = 1
NUM_EPOCHS = 2000
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
MIN_OBJECT_AREA = 0

VALIDATION_STEP = 10

# Point sampling
NUM_MAX_POINTS = 20

# Inference parameters
SAMPLE_CNT = 16

# Dataset selection
DATASET_NAME = "SkeletalMuscle"  # Choose "Lucchi" or "SkeletalMuscle"
DATASET_DIR = 'DATASET_DIR'

SAVE_DIR = 'SAVE_DIR'


# Test configuration
TEST_SAVE_DIR = 'SAVE_DIR'
TEST_MODEL_PATH = 'MODEL_PATH'
