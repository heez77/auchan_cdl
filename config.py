import os
import torch

class CFG:
    path = os.path.join(os.path.expanduser('~'), "Documents", "Auchan_Data")
    path_images = os.path.join(path, "Images")
    path_dataframe = os.path.join(path, "data.csv")
    path_models = os.path.join(path, "Models")
    path_bert = os.path.join(path, "Training", "CamemBERT")
    threshold_clip = 0.8
    threshold_dist = 0.8
    device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CFG.path, exist_ok=True)
os.makedirs(CFG.path_images, exist_ok=True)
os.makedirs(CFG.path_models, exist_ok=True)
os.makedirs(CFG.path_bert, exist_ok=True)
