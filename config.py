import torch

# Device selection
device = 'cpu'
try:
    if torch.backends.mps.is_available():
        device = 'mps'
except:
    if torch.cuda.is_available():
        device = 'cuda'

# Training parameters
PRE_TRAINED = False
PRE_TRAINED_PATH = './pi_seg/model/checkpoints/cem500k_mocov2_resnet50_200ep.pth.tar'

BATCH_SIZE = 8
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
DATASET_DIR = '../datasets/Ours/'

# 저장 경로
SAVE_DIR = './checkpoints'


# Test configuration
# 저장 경로
TEST_SAVE_DIR = './experiments'
TEST_MODEL_PATH = './pi_seg/model/checkpoints/deepPI_skeletalMuscle_deploy.pth'
