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

if __name__=='__main__':
    main()
