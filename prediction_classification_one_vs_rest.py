from os.path import dirname, abspath
import sys
import torch
from pathlib import Path
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





def get_pred_classifier(texts, i):
    version_camembert = len(os.listdir(os.path.join(CFG.path_models, 'CamemBERT_one_vs_rest')))
    DATA_PATH = os.path.join(CFG.path_bert, 'Data/')
    MODEL_PATH = Path(os.path.join(CFG.path_models, 'CamemBERT_one_vs_rest', 'CamemBERT_one_vs_rest_v{}'.format(version_camembert), 'classifier_{}'.format(i)))
    predictor = BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=DATA_PATH,  # location for labels.csv file
        multi_label=True,
        model_type='camembert-base',
        do_lower_case=False,
        device=None)
    prediction = predictor.predict_batch(texts)
    scores = [p[0][1] if p[1][0]=='Autre' else p[1][1] for p in prediction]
    return scores


def prediction_finale(texts, nb_labels, labels, treshold):
    label_list, score_list = [], []
    for i in range(nb_labels):
        true_label = labels.fr.iloc[i]
        label = [true_label, 'Autre']
        df_label = pd.DataFrame(label, columns=['label'])
        df_label.to_csv(os.path.join(CFG.path_bert, 'Data', 'labels_one_vs_rest.csv'), index=False, header=False)
        score = get_pred_classifier(texts, i)
        score_list.append(score)

    #A voir comment calculer le arg max

    for i in range(len(texts)):
        scores = []
        for j in range(nb_labels):
            scores.append(score_list[j][i])
        if max(scores)>=treshold:
            label_list.append(labels[scores.index(max(scores))])
        else:
            label_list.append('Need Human Verif')
    return label_list


def write_csv(df, labels, treshold=.8):
    texts = [text_prepare(t) for t in df.text.tolist()]
    nb_labels = len(labels)
    label_list = prediction_finale(texts, nb_labels, labels, treshold)
    df['resultat'] = label_list
    return df



def main():
    csv = glob.glob(os.path.join(CFG.path_data, 'Predictions_classification', '*.csv'))
    images = glob.glob(os.path.join(CFG.path_data, 'Predictions_classification', '*.jpg'))
    if len(csv) > 1:
        print('Trop de csv')
        exit()
    elif len(csv) == 0:
        print('Pas de csv')
        exit()
    else:
        csv = csv[0]

    df = pd.read_csv(csv, index_col=False)
    labels = pd.read_csv(CFG.path_label, index_col=False)
    df.dropna(subset=['description'], inplace=True)
    treshold = CFG.treshold
    df = write_csv(df, labels, treshold)
    df.to_csv(CFG.path_dataframe, index=False)

