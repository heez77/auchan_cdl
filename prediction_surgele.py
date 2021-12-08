from Entrainement.eff_det.eff_det_bio import EfficientDetModel, dico
import os
from config import CFG
import pandas as pd
from datetime import datetime
import torch
import PIL
import argparse

parser = argparse.ArgumentParser(description='Model version')
parser.add_argument('--version', type=int,
                    help='an integer for the version')
old_version = vars(parser.parse_args())['version']

def main(old_version=old_version):
    label = []
    version = len(os.listdir(os.path.join(CFG.path_models,'Efficient_Det_surgele')))
    if old_version== None:
        model = EfficientDetModel(num_classes=len(dico))
        MODEL_PATH = os.path.join(CFG.path_models,'Efficient_Det_bio','Efficient_Det_surgele_v{}'.format(version))
        model.load_state_dict(torch.load(MODEL_PATH))
    elif 0> old_version or old_version > version:
        print('Version invalide')
        exit()
    else:
        model = EfficientDetModel(num_classes=len(dico))
        MODEL_PATH = os.path.join(CFG.path_models,'Efficient_Det_bio','Efficient_Det_surgele_v{}'.format(old_version))
        model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    images = os.listdir(os.path.join(CFG.path_data,'Predictions_surgele'))
    IMAGES_PATH = os.path.join(CFG.path_data,'Predictions_surgele/')
    
    if len(images)==0:
        print('Aucune image à prédire')
        exit()
    else:
        imgs = [PIL.Image.open(IMAGES_PATH + image) for image in images]
        for img in imgs:
            _, _, predicted_class_labels = model.predict([img])
            if len(predicted_class_labels[0])>0:
                label.append('bio')
            else:
                label.append('non bio')
    df_resultat = pd.DataFrame(list(zip(images, label)), columns =['Image', 'Label'])
    now = datetime.now()
    date = now.strftime("%m-%d-%Y_%H%M%S")   
    OUTPUT_DIR = os.path.join(CFG.path, 'Resultats', 'Surgelé','resultat_surgele_{}.csv'.format(date))
    df_resultat.to_csv(OUTPUT_DIR)

if __name__=='__main__':
    main()