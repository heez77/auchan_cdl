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
from Data.predictions_clip.text_classification import text_prepare
from tqdm import tqdm
import warnings
warnings. simplefilter(action='ignore', category=Warning)


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

#------------------------------------------------------------------------------------------------------#


def main():
    df = pd.read_csv(CFG.path_dataframe,index_col=False)
    df_test = pd.read_csv(os.path.join(CFG.path_bert,'Data','img_test.csv'),index_col=False)
    df = df[df.image.isin(df_test.image)]
    df.dropna(subset=['description'], inplace=True)
    # df = dict({'image' : 'image.jpg',
    #            'description : 'courte description'})
    df_label = pd.read_csv(CFG.path_labels)
    # df_label = dict({'niv1' : 'Label1, Label2, ...'
    #                  'niv2' : 'Label3, Label4, ...' })
    threshold_clip = CFG.threshold_clip
    threshold_dist = CFG.threshold_dist
    df = write_csv(df, df_label, threshold_clip, threshold_dist)
    df.to_csv(CFG.path_dataframe, index=False)

def main_2():
    df = pd.read_csv(CFG.path_dataframe)
    df_label = pd.read_csv(CFG.path_labels)
    # Mesure de performances selon les thresholds
    acc_list = []
    for threshold_clip in range(np.linspace(0.0, 1.0, num=11)):
        for threshold_dist in range(np.linspace(0.0, 1.0, num=11)):
            labels = performance(df, df_label, threshold_clip, threshold_dist)
            nb = len(labels)
            accuracy = 0
            acc = []
            for i in range(nb):
                if labels(i) == df.labels[i]:
                    accuracy += 1
            acc_list.append([accuracy/nb, threshold_clip, threshold_dist])
            acc.append(accuracy/nb)

    max_idx = acc.index(max(acc))

    return acc_list[max_idx]

if __name__=='__main__':
    main()
