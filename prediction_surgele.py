from Entrainement.eff_det.eff_det_surgele import EfficientDetModel, dico
import os
from config import CFG
import pandas as pd
from datetime import datetime

def main(old_version=None):
    label = []
    if old_version== None:
        model = EfficientDetModel(num_classes=len(dico))
        version = len(os.listdir(os.path.join(CFG.path_models,'Efficient_Det_bio')))
        MODEL_PATH = os.path.join(CFG.path_models,'Efficient_Det_bio','Efficient_Det_surgele_v{}'.format(version))
        model.load_state_dict(MODEL_PATH)
    else:
        model = EfficientDetModel(num_classes=len(dico))
        MODEL_PATH = os.path.join(CFG.path_models,'Efficient_Det_bio','Efficient_Det_surgele_v{}'.format(old_version))
        model.load_state_dict(MODEL_PATH)
    images = os.listdir(os.path.join(CFG.path_data,'Predictions_surgele'))
    if len(images)==0:
        print('Aucune image à prédir')
    else:
        for image in images:
            _, _, predicted_class_labels = model.predict(image)
            if len(predicted_class_labels)>0:
                label.append('surgele')
            else:
                label.append('non surgele')
    df_resultat = pd.DataFrame(list(zip(images, label)), columns =['Image', 'Label'])
    now = datetime.now()
    date = now.strftime("%m-%d-%Y_%H%M%S")   
    OUTPUT_DIR = os.path.join(CFG.path, 'Resultats', 'Surgelé','resultat_surgele_{}.csv'.format(date))
    df_resultat.to_csv(OUTPUT_DIR)
