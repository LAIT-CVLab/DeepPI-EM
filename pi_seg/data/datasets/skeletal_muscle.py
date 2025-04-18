from pathlib import Path

import cv2
import numpy as np

from pi_seg.data.base import ISDataset
from pi_seg.data.sample import DSample


class SkeletalMuscleDataset(ISDataset):
    def __init__(self, dataset_path,
                 images_dir_name='input', masks_dir_name='target',
                 **kwargs):
        super(SkeletalMuscleDataset, self).__init__(**kwargs)
        self.name = 'Skeletal_muscle'
        self.dataset_path = Path(dataset_path)
        self._images_path = self.dataset_path / images_dir_name
        self._insts_path = self.dataset_path / masks_dir_name

        self.dataset_samples = [x.name for x in sorted(self._images_path.glob('*.tif'))]
        self.mask_samples = [x.name for x in sorted(self._insts_path.glob('*.png'))]
        self._masks_paths = {x.stem: x for x in self._insts_path.glob('*.png')}
        
    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name)

        mask_name = self.mask_samples[index]
        mask_path = str(self._masks_paths[mask_name.split('.')[0]])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        
        # convert img to numpy
        instances_mask[instances_mask > 0] = 1
        
        return DSample(image, instances_mask, objects_ids=[1], ignore_ids=[-1], sample_id=index)

    def __len__(self):
        return len(self.dataset_samples)