import warnings
warnings. simplefilter(action='ignore', category=Warning)
import pandas as pd
from formain import write_csv
from config import CFG
import numpy as np
import os


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
