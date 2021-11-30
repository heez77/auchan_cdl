import os
import torch

class CFG:
    path = os.getcwd()
    path_images = os.path.join(path, "images")
    path_dataframe = os.path.join(path, "dataframe")
    path_labels = os.path.join(path, "labels")
    path_models = os.path.join(path, )
    threshold_clip = 0.8
    threshold_dist = 0.8
    device = "cuda" if torch.cuda.is_available() else "cpu"
