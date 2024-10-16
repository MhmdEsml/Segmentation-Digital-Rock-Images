# patch_extractor.py

import numpy as np
import os
import matplotlib.pyplot as plt

def extract_patch(volume, start_d, start_h, start_w, patch_size):
    patch_d, patch_h, patch_w = patch_size
    return volume[start_d:start_d+patch_d, start_h:start_h+patch_h, start_w:start_w+patch_w]

def save_patch(patch_data, patch_seg, images_folder, masks_folder, patch_num):
    """
    Save the image and segmented mask patches to their respective folders.

    Parameters:
    - patch_data: NumPy array of the image patch.
    - patch_seg: NumPy array of the segmented mask patch.
    - images_folder: Path to the images folder.
    - masks_folder: Path to the masks folder.
    - patch_num: Patch number for naming the files.
    """
    patch_data_filename = os.path.join(images_folder, f'{patch_num:05d}.npy')
    patch_seg_filename = os.path.join(masks_folder, f'{patch_num:05d}.npy')

    np.save(patch_data_filename, patch_data)
    np.save(patch_seg_filename, patch_seg)

def create_directories(data_dirs):
    """
    Create directories for images and segmented masks.

    Parameters:
    - data_dirs: Dictionary containing paths for images, masks, and folder names.
    """
    images_dir = data_dirs["images"]
    masks_dir = data_dirs["masks"]
    folders = data_dirs["folders"]
    for folder in folders:
        os.makedirs(os.path.join(images_dir, folder), exist_ok=True)
        os.makedirs(os.path.join(masks_dir, folder), exist_ok=True)

def display_patches(patch_data, patch_seg, patch_num):
    """
    Display the image and segmented mask patches.

    Parameters:
    - patch_data: NumPy array of the image patch.
    - patch_seg: NumPy array of the segmented mask patch.
    - patch_num: Patch number for labeling.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    img_data = plt.imshow(patch_data[:, :], cmap='gray')
    plt.title(f"Patch {patch_num} - Data")
    plt.colorbar(img_data, ax=plt.gca())

    plt.subplot(1, 2, 2)
    img_seg = plt.imshow(patch_seg[:, :], cmap='gray')
    plt.title(f"Patch {patch_num} - Segmented")
    plt.colorbar(img_seg, ax=plt.gca())

    plt.show()
