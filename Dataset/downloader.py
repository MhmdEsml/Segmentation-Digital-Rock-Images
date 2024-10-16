# downloader.py

import urllib.request
from tqdm import tqdm

def download_file(url, filename):
    with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
        total_size = int(response.getheader('Content-Length').strip())
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=filename) as pbar:
            while True:
                data = response.read(1024)
                if not data:
                    break
                out_file.write(data)
                pbar.update(len(data))
