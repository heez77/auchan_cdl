import os
import urllib.request
from tqdm import tqdm
import pandas as pd

def download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = "{}.jpg".format((url.split('_0x0')[0]).split("https://media.auchan.fr/")[1])
    download_target = os.path.join(root, filename)

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return None

df = pd.read_csv("C:\\Users\\geyma\\Documents\\Centrale Digital Lab\\Projet Auchan\\données auchan\\auchan_product_media_sample.csv")

for url in df['media_url']:
    download(url, "C:\\Users\\geyma\\Documents\\Centrale Digital Lab\\Projet Auchan\\auchan_cdl\\gitignore\\données\\photos")
