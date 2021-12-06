from os.path import dirname, abspath
import sys
root_path = dirname(abspath('text_classification.py'))
sys.path.append(root_path)
import torch
import clip
from PIL import Image
import os
from config import CFG
from fast_bert.prediction import BertClassificationPredictor
import pandas as pd
import numpy as np
import os
from Training.CamemBERT.Code.text_classification import text_prepare
from tqdm import tqdm

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

def get_dist(description, model_name='model_BERT'):
    DATA_PATH = os.path.join(CFG.path_bert,'Data/')
    MODEL_PATH = os.path.join(CFG.path_models, model_name)
    predictor = BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=DATA_PATH,  # location for labels.csv file
        multi_label=True,
        model_type='camembert-base',
        do_lower_case=False,
        device=None)
    prediction = predictor.predict(text_prepare(description))
    return prediction[0][0], prediction[0][1]

def get_dist_batch(texts, model_name='model_BERT'):
    texts = [text_prepare(text) for text in texts]
    DATA_PATH = os.path.join(CFG.path_bert,'Data/')
    MODEL_PATH = os.path.join(CFG.path_models, model_name)
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

def get_clip(image, df_label, niv_tot):
    scores = []
    labels = []
    '''
    for niveau in range (1, niv_tot+1):
        if niveau == 1:
            label_clip, score_clip = simple_CLIP(os.path.join(CFG.path_images, image), df_label.niv1)
            scores.append(score_clip)
            labels.append(label_clip)   
        else :
            label_clip, score_clip = simple_CLIP(os.path.join(CFG.path_images, image), df_label['niv{}'.format(niveau)][df_label['niv{}'.format(niveau-1)] == labels[niveau-1]])
            scores.append(score_clip)
            labels.append(label_clip)
    score_clip = 1
    for score in scores:
        score_clip = score_clip * score
    '''
    label_clip, score_clip = simple_CLIP(os.path.join(CFG.path_images, image), df_label.niv2)
    return label_clip, score_clip

def write_csv(df, df_label, threshold_clip, threshold_dist):
    list_label_dist, list_score_dist = get_dist_batch(df.description.tolist(),'model_camemBERT')
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

