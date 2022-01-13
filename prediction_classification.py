from os.path import dirname, abspath
import sys
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

parser = argparse.ArgumentParser(description='Model version')
parser.add_argument('--version', type=int,
                    help='an integer for the version')
old_version = vars(parser.parse_args())['version']

def simple_CLIP(image_path, labels, model, preprocess):
    # inputs : image_path, labels (liste)
    text = clip.tokenize(labels).to(CFG.device)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(CFG.device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        prediction = logits_per_image.softmax(dim=-1).cpu().numpy()
    max_index = np.argmax(prediction)
    return (labels[max_index], prediction[0][max_index])

def get_dist(description, version_BERT):
    DATA_PATH = os.path.join(CFG.path_bert, 'Data')
    MODEL_PATH = os.path.join(CFG.path_models, 'CamemBERT', 'CamemBERT_v{}'.format(version_BERT))
    predictor = BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=DATA_PATH,  # location for labels.csv file
        multi_label=True,
        model_type='camembert-base',
        do_lower_case=False,
        device=None)
    prediction = predictor.predict(text_prepare(description))
    return prediction[0][0], prediction[0][1]

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
    return preds, scores

def get_clip(image, df_label, model, preprocess):
    label_clip, score_clip = simple_CLIP(os.path.join(CFG.path_data, 'Predictions_classification', image), df_label.en, model, preprocess)
    return label_clip, score_clip

def write_csv(df, df_label, threshold_clip, threshold_dist, version):
    print('Prédictions CamemBERT :')
    list_label_dist, list_score_dist = get_dist_batch(df.description.tolist(), version)
    print('Prédictions CLIP :')
    result = []
    model, preprocess = clip.load("ViT-B/32", device=CFG.device)
    for i in tqdm(range(len(df))):
        label_clip, score_clip = get_clip(df.image.iloc[i], df_label, model, preprocess)
        if list_label_dist[i].lower() == df_label[df_label['en']==label_clip].fr.values[0].lower():
            result.append(df_label[df_label['en']==label_clip].fr.values[0].lower())
        else:
            if score_clip > threshold_clip and list_score_dist[i] < threshold_dist :
                result.append(df_label[df_label['en']==label_clip].fr.values[0].lower())
            elif score_clip < threshold_clip and list_score_dist[i] > threshold_dist :
                result.append(list_label_dist[i])
            else:
                # Vérification humaine (API)
                result.append('Need Human Verif')
    df['resultats'] = result
    return df

#------------------------------------------------------------------------------------------------------#
def main_performance():
    csv = glob.glob(os.path.join(CFG.path_data, 'Predictions_classification', '*.csv'))
    if len(csv)>1:
        print('Trop de csv')
        exit()
    elif len(csv)==0:
        print('Pas de csv')
        exit()
    else:
        csv = csv[0]
    df = pd.read_csv(csv, index_col=False)
    df.dropna(subset=['description'], inplace=True)
    df = df[:10]
    # df = dict({'image' : 'image.jpg',
    #            'description : 'courte description'})
    df_label = pd.read_csv(CFG.path_labels)
    # df_label = dict({'niv1' : 'Label1, Label2, ...'
    #                  'niv2' : 'Label3, Label4, ...' })
    threshold_clip_list = np.linspace(0, 1, 3)
    threshold_dist_list = np.linspace(0, 1, 3)
    version = len(os.listdir(os.path.join(CFG.path_models, 'CamemBERT')))
        
    print('CamemBERT version : {}'.format(version))
    score=[]
    t_c=[]
    t_d=[]
    for threshold_clip in threshold_clip_list:
        for threshold_dist in threshold_dist_list:
            df = write_csv(df, df_label, threshold_clip, threshold_dist, version)
            vrai = 0
            tot = len(df)
            for i in range(tot):
                if df.label.iloc[i]== df.resultats.iloc[i]:
                    vrai +=1
            score.append(vrai/tot)
            t_c.append(threshold_clip)
            t_d.append(threshold_dist)
    df_perf = pd.DataFrame(list(zip(t_c,t_d,score)), columns=['treshold_CLIP', 'treshold_camemBERT', 'score'])
    df_perf.to_csv(os.path.join(CFG.path, 'Resultats', 'Classification', 'performance_BERT_v{}'.format(version)))

def main():
    csv = glob.glob(os.path.join(CFG.path_data, 'Predictions_classification', '*.csv'))
    images = glob.glob(os.path.join(CFG.path_data, 'Predictions_classification', '*.jpg'))
    if len(csv)>1:
        print('Trop de csv')
        exit()
    elif len(csv)==0:
        print('Pas de csv')
        exit()
    else:
        csv = csv[0]
    df = pd.read_csv(csv, index_col=False)
    df.dropna(subset=['description'], inplace=True)
    # df = dict({'image' : 'image.jpg',
    #            'description : 'courte description'})
    df_label = pd.read_csv(CFG.path_labels)
    # df_label = dict({'niv1' : 'Label1, Label2, ...'
    #                  'niv2' : 'Label3, Label4, ...' })
    threshold_clip = CFG.threshold_clip
    threshold_dist = CFG.threshold_dist
    version = len(os.listdir(os.path.join(CFG.path_models, 'CamemBERT')))
    if old_version==None:
        pass
    elif old_version>version or old_version<0:
        print('Version invalide')
        exit()
    else:
        version = old_version
        
    print('CamemBERT version : {}'.format(version))
    df = write_csv(df, df_label, threshold_clip, threshold_dist, version)
    now = datetime.now()
    date = now.strftime("%m-%d-%Y_%H%M%S") 
    df.to_csv(os.path.join(CFG.path, 'Resultats', 'Classification', 'resultat_classification_{}.csv'.format(date)), index=False)
    os.remove(csv)
    for image in images:
        os.remove(image)

if __name__=='__main__':
    main()
