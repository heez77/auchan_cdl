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
old_version = parser.parse_args()


def main(old_version=None):
    label = []
    if old_version== None:
        model = EfficientDetModel(num_classes=len(dico))
        version = len(os.listdir(os.path.join(CFG.path_models,'Efficient_Det_bio')))
        MODEL_PATH = os.path.join(CFG.path_models,'Efficient_Det_bio','Efficient_Det_bio_v{}'.format(version))
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        model = EfficientDetModel(num_classes=len(dico))
        MODEL_PATH = os.path.join(CFG.path_models,'Efficient_Det_bio','Efficient_Det_bio_v{}'.format(old_version))
        model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    images = os.listdir(os.path.join(CFG.path_data,'Predictions_bio'))
    IMAGES_PATH = os.path.join(CFG.path_data,'Predictions_bio/')
    
    if len(images)==0:
        print('Aucune image à prédire')
    else:
        imgs = [PIL.Image.open(IMAGES_PATH + image) for image in images]
        for i,img in enumerate(imgs):
            _, _, predicted_class_labels = model.predict([img])
            if len(predicted_class_labels[0])>0:
                label.append('bio')
            else:
                label.append('non bio')
            os.remove(os.path.join(IMAGES_PATH,images[i]))
    df_resultat = pd.DataFrame(list(zip(images, label)), columns =['Image', 'Label'])
    now = datetime.now()
    date = now.strftime("%m-%d-%Y_%H%M%S")   
    OUTPUT_DIR = os.path.join(CFG.path, 'Resultats', 'Bio','resultat_bio_{}.csv'.format(date))
    df_resultat.to_csv(OUTPUT_DIR)


if __name__=='__main__':
    main()