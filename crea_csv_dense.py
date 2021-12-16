from config import CFG
import os
import glob
import pandas as pd
from Entrainement.CamemBERT.camemBERT import text_prepare
import clip
from PIL import Image
import torch
from fast_bert.prediction import BertClassificationPredictor
from tqdm import tqdm
import numpy as np
import shutil



version = len(os.listdir(os.path.join(CFG.path_models,'CamemBERT')))
csv = glob.glob(os.path.join(CFG.path_data,'Predictions_classification','*.csv'))
images = glob.glob(os.path.join(CFG.path_data,'Predictions_classification','*.jpg'))

df_label = pd.read_csv(CFG.path_labels)

def simple_CLIP_for_dense(image_path, labels,model, preprocess):
    # inputs : image_path, labels (liste)
    text = clip.tokenize(labels).to(CFG.device)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(CFG.device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
        prediction = logits_per_image.softmax(dim=-1).cpu().numpy()
    return(prediction[0])

def get_dist_batch_for_dense(texts, version_BERT):
    texts = [text_prepare(text) for text in texts]
    prediction = []
    DATA_PATH = os.path.join(CFG.path_bert,'Data/')
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
    return prediction


def create_csv(filename, version=version, df_label=df_label):
    path = '/home/jeremy/AUCHAN 2/IMAGES/4992 images triees/avec les labels corriges'
    path_img_test = '/home/jeremy/Documents/GitHub/auchan_cdl/Entrainement/CamemBERT/Data/img_{}'.format(filename)
    path_descr_test = '/home/jeremy/Documents/GitHub/auchan_cdl/Entrainement/CamemBERT/Data/{}'.format(filename)
    path_data = '/home/jeremy/AUCHAN 2/auchan_cdl/Training/CamemBERT/Data/text_detector_data.csv'
    df_test = pd.read_csv(path_descr_test, index_col=False)
    df_img_test = pd.read_csv(path_img_test, index_col=False)
    data = pd.read_csv(path_data, index_col=False)
    data.dropna(inplace=True)
    description = [text_prepare(data['description_fournisseur'].iloc[i]+ ' ' + data['description_produit'].iloc[i]) for i in range(len(data))]
    df = df_test.copy()
    df['image'] = df_img_test.image
    del df['Unnamed: 0']
    df_result=df[['image','text']]
    df_result.rename(columns={'text': 'description'}, inplace=True)
    label = []
    columns = df.columns

    for i in range(len(df)):
        label.append(columns[df.iloc[i].values.tolist().index(1.0)])
    df_result['label'] = label
    for i in range(len(df_result)):
        df_result.image.iloc[i] = data.image.iloc[description.index(df_result.description.iloc[i])]

    df = df_result
    prediction_bert = get_dist_batch_for_dense(df.description.tolist(), version)
    df_label = pd.read_csv(CFG.path_labels)
    labels = df_label.en.tolist()
    scores_bert =[]
    pb=[]
    for pred in prediction_bert:
        scores_bert_p = np.zeros(len(labels))
        for p in pred:
            scores_bert_p[labels.index(df_label[df_label.fr==p[0]].en.values[0])] = p[1]
        scores_bert.append(list(scores_bert_p))
    scores_clip = []
    model, preprocess = clip.load("ViT-B/32", device=CFG.device)
    for i in tqdm(range(len(df))):
        a = simple_CLIP_for_dense(os.path.join(CFG.path_data,'Predictions_classification',df.image.iloc[i]), df_label.en, model, preprocess)
        scores_clip.append(a)
    entries = []
    for score_bert, score_clip in zip(scores_bert, scores_clip):
        entries.append(list(score_clip)+list(score_bert))
    print(len(entries[0]), len(entries))
    df_result = df
    df_result[[label +'_clip' for label in labels]+[label +'_bert' for label in labels]] = np.array(entries)

    df_result.to_csv(os.path.join(CFG.path,'Data','Entrainement_Dense','dense_{}'.format(filename)),index = False)

def main():
    create_csv('train.csv')
    create_csv('test.csv')
    create_csv('val.csv')

if __name__=='__main__':
    main()
