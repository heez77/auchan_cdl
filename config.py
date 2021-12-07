import os
import torch

class CFG:
    path = os.getcwd()
    path_data = os.path.join(path,'Data')
    path_dataframe = os.path.join(path, "df2.csv")
    path_labels = os.path.join(path, "df_label.csv")
    path_models = os.path.join(path, "Models")
    path_bert = os.path.join(path, 'Entrainement','CamemBERT')
    threshold_clip = 0.8
    threshold_dist = 0.8
    device = "cuda" if torch.cuda.is_available() else "cpu"
