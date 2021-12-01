import os
import torch

class CFG:
    path = os.getcwd()
    path_images = os.path.join(os.getcwd(), "gitignore/données/photos")
    path_dataframe = os.path.join(os.getcwd(), "gitignore/df.csv")
    path_labels = os.path.join(os.getcwd(), "gitignore/df_label.csv")
    threshold_clip = 0.8
    threshold_dist = 0.8
    device = "cuda" if torch.cuda.is_available() else "cpu"