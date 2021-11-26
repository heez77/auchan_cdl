import os
import torch

class CFG:
    path = os.getcwd()
    path_images = os.path.join(os.getcwd(), "images")
    path_dataframe = os.path.join(os.get_cwd(), "dataframe")
    path_labels = os.path.join(os.getcwd(), "labels")
    threshold_clip = 0.8
    threshold_dist = 0.8
    device = "cuda" if torch.cuda.is_available() else "cpu"