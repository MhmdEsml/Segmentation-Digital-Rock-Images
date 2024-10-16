# config.py

IMAGE_LINKS = {
    "Berea": {
        "number": 1,
        "original": "https://www.digitalrocksportal.org/media/projects/317/origin/1353/images/Berea_2d25um_grayscale_filtered.raw",
        "segmented": "https://www.digitalrocksportal.org/media/projects/317/origin/1354/images/Berea_2d25um_binary.raw",
        "seed": 42
    },
    "BanderaBrown": {
        "number": 2,
        "original": "https://www.digitalrocksportal.org/media/projects/317/origin/1356/images/BanderaBrown_2d25um_grayscale_filtered.raw",
        "segmented": "https://www.digitalrocksportal.org/media/projects/317/origin/1355/images/BanderaBrown_2d25um_binary.raw",
        "seed": 43
    },
    "BanderaGray": {
        "number": 3,
        "original": "https://www.digitalrocksportal.org/media/projects/317/origin/1358/images/BanderaGray_2d25um_grayscale_filtered.raw",
        "segmented": "https://www.digitalrocksportal.org/media/projects/317/origin/1357/images/BanderaGray_2d25um_binary.raw",
        "seed": 44
    },
    "Bentheimer": {
        "number": 4,
        "original": "https://www.digitalrocksportal.org/media/projects/317/origin/1361/images/Bentheimer_2d25um_grayscale_filtered.raw",
        "segmented": "https://www.digitalrocksportal.org/media/projects/317/origin/1360/images/Bentheimer_2d25um_binary.raw",
        "seed": 45
    },
    "BSG": {
        "number": 5,
        "original": "https://www.digitalrocksportal.org/media/projects/317/origin/1364/images/BSG_2d25um_grayscale_filtered.raw",
        "segmented": "https://www.digitalrocksportal.org/media/projects/317/origin/1365/images/BSG_2d25um_binary.raw",
        "seed": 46
    },
    "BUG": {
        "number": 6,
        "original": "https://www.digitalrocksportal.org/media/projects/317/origin/1367/images/BUG_2d25um_grayscale_filtered.raw",
        "segmented": "https://www.digitalrocksportal.org/media/projects/317/origin/1368/images/BUG_2d25um_binary.raw",
        "seed": 47
    },
    "BB": {
        "number": 7,
        "original": "https://www.digitalrocksportal.org/media/projects/317/origin/1370/images/BB_2d25um_grayscale_filtered.raw",
        "segmented": "https://www.digitalrocksportal.org/media/projects/317/origin/1369/images/BB_2d25um_binary.raw",
        "seed": 48
    },
    "CastleGate": {
        "number": 8,
        "original": "https://www.digitalrocksportal.org/media/projects/317/origin/1373/images/CastleGate_2d25um_grayscale_filtered.raw",
        "segmented": "https://www.digitalrocksportal.org/media/projects/317/origin/1374/images/CastleGate_2d25um_binary.raw",
        "seed": 49
    },
    "Kirby": {
        "number": 9,
        "original": "https://www.digitalrocksportal.org/media/projects/317/origin/1376/images/Kirby_2d25um_grayscale_filtered.raw",
        "segmented": "https://www.digitalrocksportal.org/media/projects/317/origin/1377/images/Kirby_2d25um_binary.raw",
        "seed": 50
    },
    "Leopard": {
        "number": 10,
        "original": "https://www.digitalrocksportal.org/media/projects/317/origin/1379/images/Leopard_2d25um_grayscale_filtered.raw",
        "segmented": "https://www.digitalrocksportal.org/media/projects/317/origin/1380/images/Leopard_2d25um_binary.raw",
        "seed": 51
    },
    "Parker": {
        "number": 11,
        "original": "https://www.digitalrocksportal.org/media/projects/317/origin/1382/images/Parker_2d25um_grayscale_filtered.raw",
        "segmented": "https://www.digitalrocksportal.org/media/projects/317/origin/1383/images/Parker_2d25um_binary.raw",
        "seed": 52
    }
}

AVAILABLE_IMAGES = list(IMAGE_LINKS.keys())

DIMENSIONS = {
    "width": 1000,
    "height": 1000,
    "depth": 1000
}



PATCH_SIZE = (128, 128, 1)

DATA_DIRS = {
    "images": "./Dataset/Data/images",
    "masks": "./Dataset/Data/segmented_masks",
    "folders": ['Training Images', 'Validation Images', 'Test Images']
}
