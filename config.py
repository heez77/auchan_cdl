import os
import torch

class CFG:
    path = os.getcwd()
    path_images = '/home/jeremy/AUCHAN 2/IMAGES/DATASET V2'
    path_dataframe = os.path.join(path, "df2.csv")
    path_labels = os.path.join(path, "df_label.csv")
    path_models = os.path.join(path, "Models")
    path_bert = os.path.join(path, 'Training/CamemBERT')
    threshold_clip = 0.8
    threshold_dist = 0.8
    device = "cuda" if torch.cuda.is_available() else "cpu"
