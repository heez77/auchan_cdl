from os.path import dirname, abspath
import sys, os
import nltk
import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
import tensorflow as tf
from fast_bert.data_cls import BertDataBunch
from fast_bert.data_lm import BertLMDataBunch
from fast_bert.learner_lm import BertLMLearner
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy, Exact_Match_Ratio
import logging
import unidecode
from pathlib import Path
root_path = dirname(abspath('text_classification.py'))
sys.path.append(root_path)
import shutil
from config import CFG
from tqdm import tqdm
import warnings
warnings. simplefilter(action='ignore', category=Warning)

#######################################################################################################################
#                      Script avec toutes les fonctions servant à l'entrainement de camemBERT.                        #
#######################################################################################################################


#       Mots et caractères à supprimer des données d'entrées

final_stopwords_list = nltk.corpus.stopwords.words('english') + nltk.corpus.stopwords.words('french') 
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
NEW_LINE = re.compile('\n')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(final_stopwords_list)

#                          Preprocessing

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()# text en minuscules
    text = unidecode.unidecode(text)
    text = NEW_LINE.sub(' ',text) # remplace les symboles NEW_LINE par un espace
    text = REPLACE_BY_SPACE_RE.sub(' ',text) # remplace les symboles REPLACE_BY_SPACE_RE par un espace
    text = BAD_SYMBOLS_RE.sub('',text) # supprime les symboles présents dans BAD_SYMBOLS_RE
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])# remplace les mots dans STOPWORDS par un espace
    return text


def preprocessing(data):
    data.dropna(inplace=True) # Supprime les entrées vides 
    labels = data.label.unique().tolist() # Récupération de la liste des labels
    X1 = data['description_fournisseur'].tolist()
    X2 = data['description_produit'].tolist()
    X = [X1[i] +' '+ X2[i] for i in range(len(X1))] # Concaténation des descriptions fournisseurs et produits
    dico = {label:i for i, label in enumerate(labels)} # Dictionnaire reliant un label à un entier
    imgs = data.index.tolist() # Liste des noms des images 
    Y = data.label.tolist() # Labels des données d'entrées 
    Y = [dico[label] for label in Y]
    Y = tf.keras.utils.to_categorical(Y, num_classes=len(dico)) # Conversion des labels en vecteur
    Y = np.array(Y)
    X = [text_prepare(text) for text in X] #Preprocessing du texte 
    X = np.array(X)

    # Séparation en un dataset d'entrainement, de validation et de test

    x, x_test, y, y_test, img, img_test = train_test_split(X,Y,imgs,test_size=0.05,train_size=0.95) 
    x_train, x_val, y_train, y_val, img_train, img_val = train_test_split(x,y,img,test_size = 0.15,train_size =0.85)

    df_lab = pd.DataFrame(columns=['label'])
    df_lab.label = labels

    df_train = pd.DataFrame(columns = ['text']+labels)
    df_val = pd.DataFrame(columns = ['text']+labels)
    df_test = pd.DataFrame(columns = ['text']+labels)

    df_img_train = pd.DataFrame(columns=['image'])
    df_img_val = pd.DataFrame(columns=['image'])
    df_img_test = pd.DataFrame(columns=['image'])

    df_train['text'] = x_train
    df_train[labels] = y_train

    df_val['text'] = x_val
    df_val[labels] = y_val

    df_test['text'] = x_test
    df_test[labels] = y_test

    df_img_train.image = img_train
    df_img_test.image = img_test
    df_img_val.image = img_val

    # Export au format csv 
    df_img_train.to_csv(os.path.join(CFG.path_bert, 'Data/img_train.csv'))
    df_img_val.to_csv(os.path.join(CFG.path_bert, 'Data/img_val.csv'))
    df_img_test.to_csv(os.path.join(CFG.path_bert, 'Data/img_test.csv'))
    df_train.to_csv(os.path.join(CFG.path_bert, 'Data/train.csv'))
    df_val.to_csv(os.path.join(CFG.path_bert,'Data/val.csv'))
    df_test.to_csv(os.path.join(CFG.path_bert,'Data/test.csv'))
    df_lab.to_csv(os.path.join(CFG.path_bert,'Data/labels.csv'), header=False, index=False)

    return X
     
#                      Création du modèle 

def model(tuning):

    DATA_PATH = Path(os.path.join(CFG.path_bert,'Data/')) # Chemin d'accès aux données d'entrainement crée pendant le preprocessing
    # Récupération des trois datasets 
    x_test = pd.read_csv('Entrainement/CamemBERT/Data/test.csv', index_col=False)
    x_train = pd.read_csv('Entrainement/CamemBERT/Data/train.csv', index_col=False)
    x_val = pd.read_csv('Entrainement/CamemBERT/Data/val.csv', index_col=False)
    texts = x_train.text.tolist() + x_test.text.tolist() + x_val.text.tolist() #Corpus entier pour le fine tuning 

    if tuning: # Fine tuning du modèle CamemBERT 
        # Création des nouveaux dossiers pour sauvegarder le modèle et les logs 
        version_fine_tuned = len(os.listdir(os.path.join(CFG.path_models, 'CamemBERT_fine_tuned')))+1
        os.mkdir(os.path.join(CFG.path_models, 'CamemBERT_fine_tuned','CamemBERT_fine_tuned_v{}'.format(version_fine_tuned)))
        os.mkdir(os.path.join(CFG.path,'Tensorboard', 'CamemBERT_fine_tuned','CamemBERT_fine_tuned_v{}'.format(version_fine_tuned)))
        os.mkdir(os.path.join(CFG.path,'Tensorboard', 'CamemBERT_fine_tuned','CamemBERT_fine_tuned_v{}'.format(version_fine_tuned),'tensorboard'))
        OUTPUT_DIR = Path(CFG.path,'Tensorboard','CamemBERT_fine_tuned','CamemBERT_fine_tuned_v{}'.format(version_fine_tuned))
        WGTS_PATH = Path(CFG.path_models,'CamemBERT_fine_tuned','CamemBERT_fine_tuned_v{}'.format(version_fine_tuned) ,'pytorch_model.bin')
        MODEL_PATH = Path(os.path.join(CFG.path_models, 'CamemBERT_fine_tuned','CamemBERT_fine_tuned_v{}'.format(version_fine_tuned)))

        logger = logging.getLogger() # Le logger enregistre les logs de l'entrainement à chaque step 

        databunch_lm = BertLMDataBunch.from_raw_corpus( # DataBunch qui charge les données d'entrainement 
                        data_dir=DATA_PATH,
                        text_list=texts, # Corpus 
                        tokenizer='camembert-base',
                        batch_size_per_gpu=4,
                        max_seq_length=512,
                        multi_gpu=False,
                        model_type='camembert-base',
                        logger=logger)
        lm_learner = BertLMLearner.from_pretrained_model( #Learner qui va s'entrainer en chargeant le modèle camemBERT pré-entrainé
                                dataBunch=databunch_lm, # Données d'entrainement 
                                pretrained_path='camembert-base', # Modèle pré-entrainé
                                output_dir=OUTPUT_DIR, # Chemin d'écriture des logs
                                metrics=[],
                                device=torch.device(CFG.device),
                                logger=logger,
                                multi_gpu=False,
                                logging_steps=50,
                                fp16_opt_level="O2")
        lm_learner.fit(epochs=30, # Entrainement sur 30 epochs
                lr=1e-4,
                validate=True,
                schedule_type="warmup_cosine",
                optimizer_type="adamw")
        lm_learner.validate()  
        lm_learner.save_model(MODEL_PATH)  # Sauvegarde du modèle 
    else: # Pas de fine tuning et chargement de la dernière version du modèle fine tuned
        version_fine_tuned = len(os.listdir(os.path.join(CFG.path_models, 'CamemBERT_fine_tuned')))
        WGTS_PATH = Path(CFG.path_models,'CamemBERT_fine_tuned','CamemBERT_fine_tuned_v{}'.format(version_fine_tuned) ,'pytorch_model.bin')
        MODEL_PATH = Path(os.path.join(CFG.path_models, 'CamemBERT_fine_tuned','CamemBERT_fine_tuned_v{}'.format(version_fine_tuned)))

    # Créations des dossiers pour l'entrainement du classifieur 
    version_camembert = len(os.listdir(os.path.join(CFG.path_models, 'CamemBERT')))+1
    os.mkdir(os.path.join(CFG.path_models, 'CamemBERT','CamemBERT_v{}'.format(version_camembert)))
    os.mkdir(os.path.join(CFG.path,'Tensorboard', 'CamemBERT','CamemBERT_v{}'.format(version_camembert)))
    BERT_PATH = Path(os.path.join(CFG.path_models, 'CamemBERT','CamemBERT_v{}'.format(version_camembert)))

    #--------------------------------------Erreurs----------------------------------#
    try:
        filename = os.listdir(os.path.join(CFG.path_data,'Entrainement_camemBERT'))[0]
    except:
        print("Aucun csv d'entrainement")
        pass
    #-------------------------------------------------------------------------------#

    OUTPUT_DIR_BERT = Path(os.path.join(CFG.path,'Tensorboard', 'CamemBERT','CamemBERT_v{}'.format(version_camembert)))
    labels = pd.read_csv(os.path.join(DATA_PATH,'labels.csv'), header=None, index_col=False)[0].tolist()
    databunch = BertDataBunch(DATA_PATH, DATA_PATH, # Référencement des paramêtres et des données
                          tokenizer='camembert-base',
                          train_file='train.csv',   
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='text', # Nom de la colonne avec les descriptions
                          label_col=labels, # Labels
                          batch_size_per_gpu=4,
                          max_seq_length=512, # Taille max d'une description
                          multi_gpu=False,
                          multi_label=True,
                          model_type='camembert-base')
    logger = logging.getLogger()
    device_cuda = torch.device("cuda") #Utilisation de la carte graphique 

    metrics = [{'name': 'Exact_Match_Ratio', 'function': Exact_Match_Ratio}] # Choix de la métrique pour le calcule de la précision du modèle 
    learner = BertLearner.from_pretrained_model( # Création du classifieur 
						databunch, # Data
						pretrained_path=MODEL_PATH, # Chemin vers le modèle camemBERT fine tuned
						metrics=metrics, # Métriques
						device=device_cuda,
						logger=logger, #Enregistreur des logs
						output_dir=OUTPUT_DIR_BERT, # Chemin d'enregistrement des logs
						finetuned_wgts_path=WGTS_PATH,
						warmup_steps=500, # Tous les 500 steps check si le gradiant n'explose pas
						multi_gpu=False, # Possibilité d'entrainer sur plusieurs GPU
						is_fp16=True,
						multi_label=True,
						logging_steps=50) # Enregistrement des logs tous les 50 steps
        
    return learner, BERT_PATH

#              Sauvegarde de la meilleure epoch par rapport à la métrique choisie 

def checkpoint(n_save, n_epochs, learner, BERT_PATH):
    match_global = 0
    early_stop=0 # Si le modèle ne s'améliore pas pendant trop longtemps, on stop l'entrainement
    lr = 9e-5
    for epoch in tqdm(range(int(n_epochs/n_save))):
        _,_,scheduler = learner.fit(epochs=n_save,
                                    lr=lr,
                                    validate=True, 	# Evalue le modèle après chaque epoch
                                    schedule_type="warmup_cosine",
                                    optimizer_type="adamw")
        scheduler.step()
        lr = scheduler.get_lr()[0] # Récupération du learning rate
        results = learner.validate()
        match = results['Exact_Match_Ratio'] # On récupére la métrique obtenue pour l'époch juste éffectuée

        if match>=match_global:  # Si la métrique s'améliore 
            learner.save_model(Path(BERT_PATH)) # Sauvegarde du modèle 
            print('Save epoch : ', str((epoch+1)*n_save))
            match_global = match
            early_stop=0
        else:
            early_stop+=n_save
            if early_stop>20:
                print("Stop epoch : ",str((epoch+1)*n_save))
                exit()
    print("Stop epoch : ",str(n_epochs))

def main_training_BERT(tuning):
    files = os.listdir(os.path.join(CFG.path_bert,'Data'))
    if 'cache' in files:
        shutil.rmtree(os.path.join(CFG.path_bert,'Data','cache'))
    learner, BERT_PATH = model(tuning)
    checkpoint(1, 200, learner, BERT_PATH) # Le premier argument détermine le nombre d'epochs a faire entre chaque sauvegarde, le second le nombre d'épochs à faire au total