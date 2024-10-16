# utils/dataset.py

import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_folder, mask_folder):
        self.image_files = sorted(glob.glob(os.path.join(image_folder, '*.npy')))
        self.mask_files = sorted(glob.glob(os.path.join(mask_folder, '*.npy')))

        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = np.load(self.image_files[idx])
        mask = np.load(self.mask_files[idx])

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize image
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # No normalization for mask (binary 0 or 1)

        return image, mask
