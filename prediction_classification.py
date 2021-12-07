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

parser = argparse.ArgumentParser(description='Model version')
parser.add_argument('--version', type=int,
                    help='an integer for the version')
old_version = vars(parser.parse_args())['version']

def simple_CLIP(image_path, labels):
    # inputs : image_path, labels (liste)
    model, preprocess = clip.load("ViT-B/32", device=CFG.device)
    text = clip.tokenize(labels).to(CFG.device)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(CFG.device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        prediction = logits_per_image.softmax(dim=-1).cpu().numpy()
    max_index = np.argmax(prediction)
    return (labels[max_index], prediction[0][max_index])

def get_dist(description, version_BERT):
    DATA_PATH = os.path.join(CFG.path_bert,'Data/')
    MODEL_PATH = os.path.join(CFG.path_models,'CamemBERT',  'CamemBERT_v{}'.format(version_BERT))
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
    DATA_PATH = os.path.join(CFG.path_bert,'Data/')
    MODEL_PATH = os.path.join(CFG.path_models,'CamemBERT',  'CamemBERT_v{}'.format(version_BERT))
    predictor = BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=DATA_PATH,  # location for labels.csv file
        multi_label=True,
        model_type='camembert-base',
        do_lower_case=False,
        device=None)
    prediction = predictor.predict_batch(texts)
    preds = [p[0][0] for p in prediction]
    scores = [p[0][1] for p in prediction]
    return preds, scores

def get_clip(image, df_label):
    scores = []
    labels = []
    label_clip, score_clip = simple_CLIP(os.path.join(CFG.path_images, image), df_label.niv2)
    return label_clip, score_clip

def write_csv(df, df_label, threshold_clip, threshold_dist, version):
    list_label_dist, list_score_dist = get_dist_batch(df.description.tolist(),version)
    for i in tqdm(range(len(df))):
        label_clip, score_clip = get_clip(df.image.iloc[i], df_label, 2)
        if list_label_dist[i][-1]=='_':
            list_label_dist[i] = list_label_dist[i][:len(list_label_dist[i])-1]
        if list_label_dist[i].lower() == df_label[df_label['niv2']==label_clip].niv2_fr.values[0].lower():
            df.result[i] = df_label[df_label['niv2']==label_clip].niv2_fr.values[0].lower()
        else:
            if score_clip > threshold_clip and list_score_dist[i] < threshold_dist :
                df.result[i] = df_label[df_label['niv2']==label_clip].niv2_fr.values[0].lower()
            elif score_clip < threshold_clip and list_score_dist[i] > threshold_dist :
                df.result[i] = list_label_dist[i]
            else:
                # Vérification humaine (API)
                df.result[i] = 'Need Human Verif'
    return df

#------------------------------------------------------------------------------------------------------#


def main():
    csv = glob.glob(os.path.join(CFG.path_data,'Predictions_classification','*.csv'))
    images = glob.glob(os.path.join(CFG.path_data,'Predictions_classification','*.jpg'))
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
    version = len(os.listdir(CFG.path_models,'CamemBERT'))
    if old_version>version or old_version<0:
        print('Version invalide')
    else:
        version = old_version
        
    
    df = write_csv(df, df_label, threshold_clip, threshold_dist, version)
    df.to_csv(CFG.path_dataframe, index=False)


if __name__=='__main__':
    main()
