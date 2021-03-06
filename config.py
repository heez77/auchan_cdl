import os
import torch

class CFG:
    path = os.getcwd()
    path_data = os.path.join(path, "Data")
    path_models = os.path.join(path, "Models")
    path_labels = os.path.join(path,'labels_en_fr.csv')
    path_bert = os.path.join(path, "Entrainement", "CamemBERT")
    path_det = os.path.join(path, "Entrainement", "eff_det")
    threshold_clip = 0.9
    threshold_dist = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"
