import torch
import clip
from PIL import Image
import os
from config import CFG
from fast_bert.prediction import BertClassificationPredictor
import pandas as pd
import numpy as np
from Entrainement.CamemBERT.camemBERT import text_prepare
from tqdm import tqdm
import warnings
warnings. simplefilter(action='ignore', category=Warning)
import argparse
import glob
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from dense import DenseModel


#######################################################################################################################
#                     Script d'execution pour la prédiction de la classification des produits.                        #
#                                                                                                                     #
#                          Ligne de commande : python3 prediction_classification.py                                   #
#           Possibilité d'ajouter l'argument --version pour choisir la version du modèle CamemBERT à choisir          #
#   Exemple pour la version 2 :                                                                                       #
#                                python3 prediction_classification.py --version 2                                     #
#######################################################################################################################



#         Argument --version pour la ligne de commande

parser = argparse.ArgumentParser(description='Model version')
parser.add_argument('--version', type=int,
                    help='an integer for the version')
old_version = vars(parser.parse_args())['version']



#          Prédiction à partir d'une image avec CLIP

def simple_CLIP(image_path, labels, model, preprocess):
    # inputs : image_path, labels (liste)
    text = clip.tokenize(labels).to(CFG.device) #On tokenize tous les labels pour que CLIP puisse faire la prédiction
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(CFG.device) # Preprocessing de l'image pour qu'elle soit sous le format comprit par CLIP
    with torch.no_grad():
        logits_per_image, _ = model(image, text) #Prédiction
        prediction = logits_per_image.softmax(dim=-1).cpu().numpy() #Analyse de la prédiction
    max_index = np.argmax(prediction) #Récupération du meilleur score
    return (labels[max_index], prediction[0][max_index]) #Renvoie de la meilleure prédiction et de son score

#         Prédiction à partir d'une description avec CamemBERT

def get_dist(description, version_BERT):
    DATA_PATH = os.path.join(CFG.path_bert, 'Data') #Chemin d'accès aux données
    MODEL_PATH = os.path.join(CFG.path_models, 'CamemBERT', 'CamemBERT_v{}'.format(version_BERT)) #CHemin d'accès au modèle de classification camemBERT
    predictor = BertClassificationPredictor( #Chargement d'un classificateur à partir du modèle entrainé
        model_path=MODEL_PATH,
        label_path=DATA_PATH,  # location for labels.csv file
        multi_label=True,
        model_type='camembert-base',
        do_lower_case=False,
        device=None)
    prediction = predictor.predict(text_prepare(description)) #Prédiction
    return prediction[0][0], prediction[0][1] #Renvoie de la meilleure prédiction avec son score

#         Prédiction à partir d'une liste de déscription avec camemBERT
#                       Similaire à la fonction d'avant 

def get_dist_batch(texts, version_BERT):    
    texts = [text_prepare(text) for text in texts]
    prediction = []
    DATA_PATH = os.path.join(CFG.path_bert, 'Data')
    MODEL_PATH = os.path.join(CFG.path_models,'CamemBERT',  'CamemBERT_v{}'.format(version_BERT))
    predictor = BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=DATA_PATH,  # location for labels.csv file
        multi_label=True,
        model_type='camembert-base',
        do_lower_case=False,
        device=None)
    for text in tqdm(texts):
        prediction.append(predictor.predict(text))
    preds = [p[0][0] for p in prediction]
    scores = [p[0][1] for p in prediction]
    return preds, scores # Renvoie la liste des meilleurs prédictions et leurs scores

#           Fonction pour éffectuer la prédiction CLIP

def get_clip(image, df_label, model, preprocess):
    label_clip, score_clip = simple_CLIP(os.path.join(CFG.path_data, 'Predictions_classification', image), df_label.en, model, preprocess)
    return label_clip, score_clip

#          Fonction qui récupère les données effectue les prédictions et écrit un DataFrame avec les résultats

def write_csv(df, df_label, threshold_clip, threshold_dist, version):
    print('Prédictions CamemBERT :')
    list_label_dist, list_score_dist = get_dist_batch(df.description.tolist(), version) #Prédictions CamemBERT
    labels = df_label.en.tolist()
    scores_bert =[]
    scores_clip=[]
    for pred in list_label_dist:
        scores_bert_p = np.zeros(len(labels))
        for p in pred:
            scores_bert_p[labels.index(df_label[df_label.fr==p[0]].en.values[0])] = p[1]
        scores_bert.append(list(scores_bert_p))
    print('Prédictions CLIP :')
    result = []
    model, preprocess = clip.load("ViT-B/32", device=CFG.device) #Chargement du modèle CLIP
    for i in tqdm(range(len(df))): #On boucle sur tous les produits à classifier 
        label_clip, score_clip = get_clip(df.image.iloc[i], df_label, model, preprocess) #Prédiction CLIP pour une image
        scores_clip.append(score_clip)

    input = np.array([scores_clip[i] + scores_bert[i] for i in range(len(scores_clip))])
    dense = DenseModel()
    dense.load_state_dict(torch.load(os.path.join(CFG.path_models,'Dense'))) #Chargement du modèle dense
    res = dense.predict(input) # Prédictions couche dense
    result = []
    for r in res:
        idx = r.index(max(r))
        result.append(labels[idx])

    df['resultats'] = result
    return df #Renvoie la dataframe avec les résultas

#------------------------------------------------------------------------------------------------------#


def main():
    csv = glob.glob(os.path.join(CFG.path_data, 'Predictions_classification', '*.csv')) #Récupération du csv avec les descriptions
    images = glob.glob(os.path.join(CFG.path_data, 'Predictions_classification', '*.jpg')) #Récupération des images

    #-------Erreurs-------#
    if len(csv)>1:
        print('Trop de csv')
        exit()
    elif len(csv)==0:
        print('Pas de csv')
        exit()
    #---------------------#

    else:
        csv = csv[0]
    df = pd.read_csv(csv, index_col=False)[:10]
    df.dropna(subset=['description'], inplace=True) #Suppression des produits sans description
    df_label = pd.read_csv(CFG.path_labels) #Récupération des labels
    threshold_clip = CFG.threshold_clip
    threshold_dist = CFG.threshold_dist
    version = len(os.listdir(os.path.join(CFG.path_models, 'CamemBERT')))

    #--------Version--------#
    if old_version==None:
        pass
    elif old_version>version or old_version<0:
        print('Version invalide')
        exit()
    else:
        version = old_version
    #-----------------------#
        
    print('CamemBERT version : {}'.format(version))
    df = write_csv(df, df_label, threshold_clip, threshold_dist, version) #DataFrame avec les résultats des prédictions
    now = datetime.now()
    date = now.strftime("%m-%d-%Y_%H%M%S")  #Récupération de la date de prédiction
    df.to_csv(os.path.join(CFG.path, 'Resultats', 'Classification', 'resultat_classification_{}.csv'.format(date)), index=False) #Enregistrement en csv 
    # sous le format resultat_classification_DateDePrediction.csv

    #---Supression données de prédictions---#
    os.remove(csv)
    for image in images:
        os.remove(image)
    #---------------------------------------#

if __name__=='__main__':
    main()
