import torch
import clip
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

list_directories = ["beer or wine", "alcohol"]

text = clip.tokenize(list_directories).to(device)

image = preprocess(Image.open(os.path.join(os.getcwd(), "heineken-25-cl.jpg"))).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    prediction = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Predicted label:", prediction)
