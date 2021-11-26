from clip_dist import CLIP, DIST

def get_dist():
    return

def get_clip(image, df_label, niv_tot):
    scores = []
    labels = []
    for niveau in range (1, niv_tot+1):
        if niveau == 1:
            label_clip, score_clip = CLIP(image, df_label.niv1)
            scores.append(score_clip)
            labels.append(label_clip)
        else :
            label_clip, score_clip = CLIP(image, df_label['niv{}'.format(niveau)][df_label['niv{}'.format(niveau-1)] == labels[niveau-1]])
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