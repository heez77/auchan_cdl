import os
import urllib.request
from tqdm import tqdm
import pandas as pd
import ssl

def download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = "{}.jpg".format((url.split('_0x0')[0]).split("https://media.auchan.fr/")[1])
    download_target = os.path.join(root, filename)


    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return None

csv_path = os.path.join("c:\\", "Users", "Tijoxa", "Downloads", "data", "auchan_product_media_sample.csv")
download_path = os.path.join("c:\\", "Users", "Tijoxa", "Documents", "GitHub", "dataset")

df = pd.read_csv(csv_path)

for url in df['media_url']:
    download(url, download_path)
