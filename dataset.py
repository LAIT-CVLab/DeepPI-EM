import os
from torch.utils.data import DataLoader
from torchvision import transforms as torch_transforms

from pi_seg.utils.exp_imports.default import *
from pi_seg.data.datasets.lucchi import LucchiDataset
from pi_seg.data.datasets.skeletal_muscle import SkeletalMuscleDataset
from pi_seg.data.transforms import *

from config import MIN_OBJECT_AREA, DATASET_NAME, DATASET_DIR, BATCH_SIZE, VAL_BATCH_SIZE, NUM_MAX_POINTS


# points sampler 
points_sampler = MultiPointSampler(NUM_MAX_POINTS, prob_gamma=0.80,
                                   merge_objects_prob=0.15,
                                   max_num_merged_objects=2,
                                   use_hierarchy=False,
                                   first_click_center=False)

# Define data transformations
transform_train = torch_transforms.Compose([ToTensor(), Resize(), RandomCrop(), RandomFlip(), Rotate()])
transform_val = torch_transforms.Compose([ToTensor(), Resize()])

# Select dataset dynamically
if DATASET_NAME == "Lucchi":
    DatasetClass = LucchiDataset
elif DATASET_NAME == "SkeletalMuscle":
    DatasetClass = SkeletalMuscleDataset
else:
    raise ValueError("Invalid dataset name! Choose between 'Lucchi' and 'SkeletalMuscle'.")

# Training dataset
trainset = DatasetClass(
    dataset_path=os.path.join(DATASET_DIR, 'train'),
    images_dir_name='input',
    masks_dir_name='target',
    augmentator=transform_train,
    min_object_area=MIN_OBJECT_AREA,
    keep_background_prob=0.05,
    points_sampler=points_sampler,
)

# Validation dataset
valset = DatasetClass(
    dataset_path=os.path.join(DATASET_DIR, 'test'),
    images_dir_name='input',
    masks_dir_name='target',
    augmentator=transform_val,
    min_object_area=MIN_OBJECT_AREA,
    points_sampler=points_sampler,
)

# DataLoaders
train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=8)
val_dataloader = DataLoader(valset, batch_size=VAL_BATCH_SIZE, pin_memory=True, num_workers=8)


# --- Display Dataset Information ---
print("\n" + "="*50)
print("DATASET & DATALOADER INFORMATION")
print("="*50)
print(f"[*] Dataset Name : {DATASET_NAME}")
print(f"[*] Dataset Path : {DATASET_DIR}")
# -----------------------------------


# --- Display Dataloader Information ---
print(f"[*] Train Samples      : {len(trainset)} images, {len(train_dataloader)} batches")
print(f"[*] Train Batch Size   : {BATCH_SIZE}")
print(f"[*] Validation Samples : {len(valset)} images, {len(val_dataloader)} batches")
print(f"[*] Validation Batch Size: {VAL_BATCH_SIZE}")
print("="*50 + "\n")
# ------------------------------------