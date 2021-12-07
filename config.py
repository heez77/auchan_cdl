import os
import torch

class CFG:
    path = os.getcwd()
    path_data = os.path.join(path, "Data")
    path_models = os.path.join(path, "Models")
    path_labels = os.path.join(path,'df_label.csv')
    path_bert = os.path.join(path, "Entrainement", "CamemBERT")
    path_det = os.path.join(path, "Entrainement", "eff_det")
    threshold_clip = 0.8
    threshold_dist = 0.8
    device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CFG.path_data, exist_ok=True)
os.makedirs(CFG.path_models, exist_ok=True)
