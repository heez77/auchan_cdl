import torch
import clip
from PIL import Image
import os
from config import CFG
from fast_bert.prediction import BertClassificationPredictor
import pandas as pd

def simple_CLIP(image_path, labels):
    # inputs : image_path, labels (liste)
    model, preprocess = clip.load("ViT-B/32", device=CFG.device)
    text = clip.tokenize(labels).to(CFG.device)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(CFG.device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        prediction = logits_per_image.softmax(dim=-1).cpu().numpy()
    max_value = max(prediction)
    max_index = prediction.index(max_value)
    return (labels[max_index], max_value)

def get_dist(description, model_name='bert.bin'):
    DATA_PATH = '/home/jeremy/Documents/GitHub/auchan_cdl/CamemBERT/Data/'
    MODEL_PATH = os.path.join(CFG.path_models, model_name)
    labels = pd.read_csv(os.path.join(DATA_PATH, 'labels.csv'), header=None, index_col=False)[0].tolist()
    predictor = BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=DATA_PATH,  # location for labels.csv file
        multi_label=True,
        model_type='bert',
        do_lower_case=False,
        device=None)
    prediction = predictor.predict(description)
    return prediction

def get_clip(image, df_label, niv_tot):
    scores = []
    labels = []
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
    return (labels[-1], score_clip)

def write_csv(df, df_label, threshold_clip, threshold_dist):
    for i in range (len(df)):
        label_dist, score_dist = get_dist(df.description[i])
        label_clip, score_clip = get_clip(df.image[i], df_label, 2)
        if label_dist == label_clip:
            df.result[i] = label_clip
        else:
            if score_clip > threshold_clip and score_dist < threshold_dist :
                df.result[i] = label_clip
            elif score_clip < threshold_clip and score_dist > threshold_dist :
                df.result[i] = label_dist
            else :
                # Vérification humaine (API)
                df.result[i] = 'Need Human Verif'