import numpy as np
import pandas as pd
import os

path = '/home/jeremy/AUCHAN 2/IMAGES/RESULTS DATASET V2'

def get_labels(path=path):
    image = []
    label = []
    filenames = os.listdir(path)
    for folder in filenames:
        imgs = os.listdir(os.path.join(path, folder))
        for img in imgs:
            image.append(img)
            label.append(folder)
    df = pd.DataFrame(columns=['image', 'label'])
    df.image = image
    df.label = label
    return df