import pandas as pd
from formain import write_csv
from config import CFG

def main():
    df = pd.read_csv(CFG.path_dataframe)
    # df = dict({'image' : 'image.jpg',
    #            'description : 'courte description'})
    df_label = pd.read_csv(CFG.path_labels)
    # df_label = dict({'niv1' : 'Label1, Label2, ...'
    #                  'niv2' : 'Label3, Label4, ...' })
    threshold_clip = CFG.threshold_clip
    threshold_dist = CFG.threshold_dist
    write_csv(df, df_label, threshold_clip, threshold_dist)

def main_2():
    df = pd.read_csv(CFG.path_dataframe)
    df_label = pd.read_csv(CFG.path_labels)
    # Mesure de performances selon les thresholds
    for threshold_clip in range(np.linspace(0.0, 1.0, num=11)):
        for threshold_dist in range(np.linspace(0.0, 1.0, num=11)):
            labels = performance(df, df_label, threshold_clip, threshold_dist)

    nb = len(labels)
    accuracy = 0
    for i in range(nb):
        if labels(i) == df.labels[i]:
            accuracy += 1
    return accuracy/nb

if __name__=='__main__':
    main()
