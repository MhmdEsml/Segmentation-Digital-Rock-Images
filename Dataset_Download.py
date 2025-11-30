# main.py

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from Dataset.config import IMAGE_LINKS, AVAILABLE_IMAGES, DIMENSIONS, PATCH_SIZE, DATA_DIRS
from Dataset.downloader import download_file
from Dataset.patch_extractor import extract_patch, save_patch, create_directories, display_patches
from Dataset.utils import load_raw_file
from tqdm import tqdm


def main():
    print("Available images:", ", ".join(AVAILABLE_IMAGES))
    selected_image = input("Please enter the image you want to download (e.g., Berea): ")
    num_train_patches = int(input("Please enter the number of train patches: "))
    num_val_patches = int(input("Please enter the number of validation patches: "))
    num_test_patches = int(input("Please enter the number of test patches: "))

    if selected_image not in IMAGE_LINKS:
        raise ValueError(f"Invalid image selected: {selected_image}")

    n_image = int(IMAGE_LINKS[selected_image]["number"])
    random_seed = IMAGE_LINKS[selected_image]["seed"]

    original_url = IMAGE_LINKS[selected_image]["original"]
    segmented_url = IMAGE_LINKS[selected_image]["segmented"]

    print(f"Downloading {selected_image} original data...")
    download_file(
        selected_image,
        f'{selected_image}_2d25um_grayscale_filtered.raw'
    )

    print(f"Downloading {selected_image} segmented data...")
    download_file(
        f'{selected_image}_binary',
        f'{selected_image}_2d25um_binary.raw'
    )

    width = DIMENSIONS["width"]
    height = DIMENSIONS["height"]
    depth = DIMENSIONS["depth"]

    data = load_raw_file(f'{selected_image}_2d25um_grayscale_filtered.raw', (depth, height, width))
    data_seg = load_raw_file(f'{selected_image}_2d25um_binary.raw', (depth, height, width))

    print("Dimensions of the original file:", data.shape)
    print("Dimensions of the segmented file:", data_seg.shape)

    total_patches = num_train_patches + num_val_patches + num_test_patches

    create_directories(DATA_DIRS)

    random.seed(random_seed)
    for i in tqdm(range(1, total_patches + 1)):
        d, h, w = data.shape
        patch_d, patch_h, patch_w = PATCH_SIZE
        start_d = random.randint(0, d - patch_d)
        start_h = random.randint(0, h - patch_h)
        start_w = random.randint(0, w - patch_w)

        patch_data = extract_patch(data, start_d, start_h, start_w, PATCH_SIZE).squeeze()
        patch_seg = extract_patch(data_seg, start_d, start_h, start_w, PATCH_SIZE).squeeze()

        if i <= num_train_patches:
            folder = 'Training Images'
            patch_num = i + int(num_train_patches * (n_image - 1))
        elif i <= num_train_patches + num_val_patches:
            folder = 'Validation Images'
            patch_num = i - num_train_patches + int(num_val_patches * (n_image - 1))
        else:
            folder = 'Test Images'
            patch_num = i - num_train_patches - num_val_patches + int(num_test_patches * (n_image - 1))

        images_folder = os.path.join(DATA_DIRS["images"], folder)
        masks_folder = os.path.join(DATA_DIRS["masks"], folder)

        save_patch(patch_data, patch_seg, images_folder, masks_folder, patch_num)

        if i <= 5:
            display_patches(patch_data, patch_seg, i)

    print(f"Extracted and saved {num_train_patches} training patches, {num_val_patches} validation patches, "
          f"and {num_test_patches} test patches in the respective folders.")

if __name__ == "__main__":
    main()


