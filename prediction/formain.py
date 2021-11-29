import torch
import clip
from PIL import Image
import os
from prediction.config import CFG
import numpy as np

def simple_CLIP(image, labels):
    # inputs : image_path, labels (liste)
    model, preprocess = clip.load("ViT-B/32", device=CFG.device)
    text = clip.tokenize(labels).to(CFG.device)
    image = preprocess(Image.open(image)).unsqueeze(0).to(CFG.device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        prediction = logits_per_image.softmax(dim=-1).cpu().numpy()
    max_value = max(prediction[0])
    max_index = np.where(prediction[0] == max_value)
    return (labels[max_index[0][0]], max_value)

def simple_DIST(model, description):
    with torch.no_grad():
        model.eval()
        input_ids, attention_mask = preprocess(reviews)
        retour = model(input_ids, attention_mask = attention_mask)

def get_clip(image, df_label, niv_tot):
    scores = []
    labels = []
    for niveau in range (1, niv_tot+1):
        if niveau == 1:
            label_clip, score_clip = simple_CLIP(image, df_label.niv1.tolist())
            scores.append(score_clip)
            labels.append(label_clip)
        else :
            index_pre_niv = df_label[df_label['niv{}'.format(niveau-1)]==labels[niveau-1-1]].index.tolist()
            label_clip, score_clip = simple_CLIP(image, df_label['niv{}'.format(niveau)][index_pre_niv].tolist())
            scores.append(score_clip)
            labels.append(label_clip)
    score_clip = 1
    for score in scores:
        score_clip *= score
    return (labels[-1], score_clip)

def write_csv(df, df_label, threshold_clip, threshold_dist):
    for i in range (len(df)):
        image_path = "../gitignore/données/photos/{}.jpeg".format(df.image[i])
        label_dist, score_dist = simple_DIST(df.description[i])
        label_clip, score_clip = get_clip(image_path, df_label, 2)
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