import os
import urllib.parse
import urllib.request
from tqdm import tqdm


POROUS_MEDIA_DATASETS = {
    "Berea": "Berea/Berea_2d25um_grayscale_filtered.raw/Berea_2d25um_grayscale_filtered.raw",
    "Berea_binary": "Berea/Berea_2d25um_binary.raw/Berea_2d25um_binary.raw",
    "BanderaBrown": "Bandera Brown/BanderaBrown_2d25um_grayscale_filtered.raw/BanderaBrown_2d25um_grayscale_filtered.raw",
    "BanderaBrown_binary": "Bandera Brown/BanderaBrown_2d25um_binary.raw/BanderaBrown_2d25um_binary.raw",
    "BanderaGray": "Bandera Gray/BanderaGray_2d25um_grayscale_filtered.raw/BanderaGray_2d25um_grayscale_filtered.raw",
    "BanderaGray_binary": "Bandera Gray/BanderaGray_2d25um_binary.raw/BanderaGray_2d25um_binary.raw",
    "Bentheimer": "Bentheimer/Bentheimer_2d25um_grayscale_filtered.raw/Bentheimer_2d25um_grayscale_filtered.raw",
    "Bentheimer_binary": "Bentheimer/Bentheimer_2d25um_binary.raw/Bentheimer_2d25um_binary.raw",
    "BSG": "Berea Sister Gray/BSG_2d25um_grayscale_filtered.raw/BSG_2d25um_grayscale_filtered.raw",
    "BSG_binary": "Berea Sister Gray/BSG_2d25um_binary.raw/BSG_2d25um_binary.raw",
    "BUG": "Berea Upper Gray/BUG_2d25um_grayscale_filtered.raw/BUG_2d25um_grayscale_filtered.raw",
    "BUG_binary": "Berea Upper Gray/BUG_2d25um_binary.raw/BUG_2d25um_binary.raw",
    "BuffBerea": "Buff Berea/BB_2d25um_grayscale_filtered.raw/BB_2d25um_grayscale_filtered.raw",
    "BuffBerea_binary": "Buff Berea/BB_2d25um_binary.raw/BB_2d25um_binary.raw",
    "CastleGate": "CastleGate/CastleGate_2d25um_grayscale_filtered.raw/CastleGate_2d25um_grayscale_filtered.raw",
    "CastleGate_binary": "CastleGate/CastleGate_2d25um_binary.raw/CastleGate_2d25um_binary.raw",
    "Kirby": "Kirby/Kirby_2d25um_grayscale_filtered.raw/Kirby_2d25um_grayscale_filtered.raw",
    "Kirby_binary": "Kirby/Kirby_2d25um_binary.raw/Kirby_2d25um_binary.raw",
    "Leopard": "Leopard/Leopard_2d25um_grayscale_filtered.raw/Leopard_2d25um_grayscale_filtered.raw",
    "Leopard_binary": "Leopard/Leopard_2d25um_binary.raw/Leopard_2d25um_binary.raw",
    "Parker": "Parker/Parker_2d25um_grayscale_filtered.raw/Parker_2d25um_grayscale_filtered.raw",
    "Parker_binary": "Parker/Parker_2d25um_binary.raw/Parker_2d25um_binary.raw",
}

DATASET_BASE_URL = "https://web.corral.tacc.utexas.edu/digitalporousmedia/DRP-317"

def download_file(url_key: str, output_path: str, force_download: bool = False):
    if url_key not in POROUS_MEDIA_DATASETS:
        raise ValueError(f"Unknown dataset key: {url_key}")

    if os.path.exists(output_path) and not force_download:
        return

    relative_path = POROUS_MEDIA_DATASETS[url_key]
    encoded_path = "/".join(urllib.parse.quote(p) for p in relative_path.split("/"))
    download_url = f"{DATASET_BASE_URL}/{encoded_path}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=os.path.basename(output_path)) as pbar:
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                pbar.total = total_size
            pbar.update(block_size)

        urllib.request.urlretrieve(download_url, output_path, reporthook=reporthook)



