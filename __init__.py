"""
Create directories and subdirectories
"""

import os

root_path = os.getcwd()

os.makedirs(os.path.join(root_path, "Data"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Data", "Entrainement_camemBERT"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Data", "Predictions_classification"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Data", "Predictions_surgelés"), exist_ok=True)

os.makedirs(os.path.join(root_path, "Models"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Models", "CamemBERT"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Models", "Efficient_Det_surgele"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Models", "Fine_Tuned_BERT"), exist_ok=True)

os.makedirs(os.path.join(root_path, "Resultats"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Resultats", "Bio"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Resultats", "Classification"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Resultats", "Surgelé"), exist_ok=True)

os.makedirs(os.path.join(root_path, "Tensorboard"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Tensorboard", "CamemBERT"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Tensorboard", "CamemBERT_fine_tuned"), exist_ok=True)
os.makedirs(os.path.join(root_path, "Tensorboard", "Efficient_Det_bio"), exist_ok=True)
