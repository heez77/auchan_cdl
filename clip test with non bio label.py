import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

list_directories = os.listdir(os.getcwd() + "\\Documents\\GitHub\\auchan_cdl\\gitignore\\NON BIO LABEL")

text = clip.tokenize(list_directories).to(device)

for folder in list_directories:
    for file in os.listdir(os.getcwd() + "\\Documents\\GitHub\\auchan_cdl\\gitignore\\NON BIO LABEL\\" + folder):
        image = preprocess(Image.open(os.getcwd() + "\\Documents\\GitHub\\auchan_cdl\\gitignore\\NON BIO LABEL\\" + folder + "\\" + file)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            probs = list(probs[0])
            max_value = max(probs)
            max_index = probs.index(max_value)
            prediction = list_directories[max_index]

        print("Predicted label:", prediction, "\nReal label:", folder, "\n")
