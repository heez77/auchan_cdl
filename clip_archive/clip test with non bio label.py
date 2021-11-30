import torch
import clip
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
list_directories = pd.read_csv('/home/jeremy/Documents/GitHub/auchan_cdl/clip_archive/Labels.csv', header=None)[3].tolist()

list_directories = list_directories[:list_directories.index(np.nan)]
predictions =[]
images=[]
score=[]
text = clip.tokenize(list_directories).to(device)
filenames = os.listdir("/home/jeremy/AUCHAN 2/IMAGES/DATASET V2/")
for file in tqdm(filenames):
    image = preprocess(Image.open("/home/jeremy/AUCHAN 2/IMAGES/DATASET V2/" + file)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        probs = list(probs[0])
        max_value = max(probs)
        max_index = probs.index(max_value)
        prediction = list_directories[max_index]
        images.append(file)
        predictions.append(prediction)
        score.append(max_value)

df = pd.DataFrame(columns=['image', 'prediction', 'score'])
df.image = images
df.prediction = predictions
df.score = score

df.to_csv('predictions_w_clip.csv')
