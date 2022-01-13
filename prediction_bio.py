from Entrainement.eff_det.eff_det_bio import EfficientDetModel, dico
import os
from config import CFG
import pandas as pd
from datetime import datetime
import torch
import PIL
import argparse

#######################################################################################################################
#                     Script d'execution pour la prédiction de la classification des produits.                        #
#                                                                                                                     #
#                                Ligne de commande : python3 prediction_bio.py                                        #
#            Possibilité d'ajouter l'argument --version pour choisir la version du modèle EffDet à choisir            #
#   Exemple pour la version 2 :                                                                                       #
#                                       python3 prediction_bio.py --version 2                                         #
#######################################################################################################################



#         Argument --version pour la ligne de commande

parser = argparse.ArgumentParser(description='Model version')
parser.add_argument('--version', type=int,
                    help='an integer for the version')
old_version = vars(parser.parse_args())['version']

#---------------------------------------------------------------------------------------------------------------------#

def main(old_version=old_version):
    label = []
    # Choix de la version
    version = len(os.listdir(os.path.join(CFG.path_models,'Efficient_Det_bio')))
    if old_version== None:
        model = EfficientDetModel(num_classes=len(dico)) 
        MODEL_PATH = os.path.join(CFG.path_models,'Efficient_Det_bio','Efficient_Det_bio_v{}'.format(version))
        model.load_state_dict(torch.load(MODEL_PATH))
    elif 0> old_version or old_version > version:
        print('Version invalide')
        exit()
    else:
        model = EfficientDetModel(num_classes=len(dico))
        MODEL_PATH = os.path.join(CFG.path_models,'Efficient_Det_bio','Efficient_Det_bio_v{}'.format(old_version))
        model.load_state_dict(torch.load(MODEL_PATH))
    
    model.eval() # Initialisation du modèle
    images = os.listdir(os.path.join(CFG.path_data,'Predictions_bio')) # Liste des images à prédire
    IMAGES_PATH = os.path.join(CFG.path_data,'Predictions_bio/') # Chemin d'accès aux images
    
    if len(images)==0:
        print('Aucune image à prédire')
        exit()
    else:
        imgs = [PIL.Image.open(IMAGES_PATH + image) for image in images] # Chargement des images
        for i,img in enumerate(imgs):
            _, _, predicted_class_labels = model.predict([img]) # Prediction
            # Classification 
            if len(predicted_class_labels[0])>0: 
                label.append('bio')
            else:
                label.append('non bio')
            os.remove(os.path.join(IMAGES_PATH,images[i])) # Suppression de l'image
    df_resultat = pd.DataFrame(list(zip(images, label)), columns =['Image', 'Label']) # Ecriture des résultats
    now = datetime.now()
    date = now.strftime("%m-%d-%Y_%H%M%S")   
    OUTPUT_DIR = os.path.join(CFG.path, 'Resultats', 'Bio','resultat_bio_{}.csv'.format(date)) # Chemin d'accès des résultqts de la prédiction
    df_resultat.to_csv(OUTPUT_DIR) # csv sous format resultat_bio_DateDePrediction.csv 


if __name__=='__main__':
    main()