import pandas as pd
import os

df = pd.read_csv("gitignore/df.csv")

# Pour Clip directement au niveau 2
prediction_clip = df
prediction_clip = prediction_clip[['image_name_MEDIASTEP','image_name','label_niv2']]
labels = [] # couple est une liste des correct_label dans l'ordre
for image in prediction_clip.image_name:
    index = (prediction_clip[prediction_clip.image_name==image].index.values)[0]
    # Trouver le folder où est image    
    for folder in os.listdir("gitignore/RESULTS DATASET V2/RESULTS DATASET V2"):
        if '{}.jpg'.format(image) in os.listdir("gitignore/RESULTS DATASET V2/RESULTS DATASET V2/" + folder):
            prediction_clip.label_niv2[index]=folder
            print('image n°{} done'.format(index))
prediction_clip.to_csv("gitignore/prediction_clip.csv")

# Pour Clip step by step
prediction_clip_step = df
prediction_clip_step = prediction_clip[['image_name_MEDIASTEP','image_name','label']]
labels = [] # couple est une liste des correct_label dans l'ordre
for image in prediction_clip_step.image_name:
    index = (prediction_clip_step[prediction_clip_step.image_name==image].index.values)[0]
    # Trouver le folder où est image
    for folder in os.listdir("gitignore/RESULTS DATASET V2 STEP/RESULTS DATASET V2 STEP"):
        if '{}.jpg'.format(image) in os.listdir("gitignore/RESULTS DATASET V2 STEP/RESULTS DATASET V2 STEP/" + folder):
            prediction_clip_step.label[index]=folder
            print('image n°{} done'.format(index))
prediction_clip_step.to_csv("gitignore/prediction_clip_step.csv")

# Comparer les deux colonnes label des deux nouveaux dataset avec la colonne label de df
import pandas as pd
import os
import numpy as np

df = pd.read_csv("gitignore/df.csv")
df = df.reindex(np.linspace(0,len(df)-1,len(df)).astype(int).tolist())
df = df[['image_name_MEDIASTEP','image_name', 'description', 'label', 'media_url']]
for image in df.image_name:
    index = (df[df.image_name==image].index.values)[0]
    # Trouver le folder où est image    
    for folder in os.listdir("gitignore/Correct_labels"):
        if '{}.jpg'.format(image) in os.listdir("gitignore/Correct_labels/" + folder):
            df.label[index]=folder
            print('image n°{} done'.format(index))
df.to_csv("gitignore/df.csv")




import pandas as pd
import os
import numpy as np

df = pd.read_csv("gitignore/df.csv")
df_label = pd.read_csv("gitignore/df_label.csv")
for i in range(len(df)):
    index = (df_label[df_label.niv2_fr==df.label_niv2[i]].index.values)[0]
    df['label_niv1'][i] = df_label.niv1_fr[index]
df = df[['image_name_MEDIASTEP','image_name', 'description', 'label_niv1', 'label_niv2', 'media_url']]
df.to_csv("gitignore/df.csv")

import pandas as pd
import os
import numpy as np
prediction_clip = pd.read_csv("gitignore/prediction_clip.csv")
df_label = pd.read_csv("gitignore/df_label.csv")
prediction_clip['label_niv1'] = 0
for i in range(len(prediction_clip)):
    index = (df_label[df_label.niv2_fr==prediction_clip.label_niv2[i]].index.values)[0]
    prediction_clip['label_niv1'][i] = df_label.niv1_fr[index]
prediction_clip = prediction_clip[['image_name_MEDIASTEP','image_name', 'label_niv1', 'label_niv2']]
prediction_clip.to_csv("gitignore/prediction_clip.csv")

import pandas as pd
import os
import numpy as np
prediction_clip_step = pd.read_csv("gitignore/prediction_clip_step.csv")
df_label = pd.read_csv("gitignore/df_label.csv")
prediction_clip_step['label_niv1'] = 0
for i in range(len(prediction_clip_step)):
    index = (df_label[df_label.niv2_fr==prediction_clip_step.label_niv2[i]].index.values)[0]
    prediction_clip_step['label_niv1'][i] = df_label.niv1_fr[index]
prediction_clip_step = prediction_clip_step[['image_name_MEDIASTEP','image_name', 'label_niv1', 'label_niv2']]
prediction_clip_step.to_csv("gitignore/prediction_clip_step.csv")

import pandas as pd
import os
import numpy as np

df = pd.read_csv("gitignore/df.csv")
df_label = pd.read_csv("gitignore/df_label.csv")
prediction_clip = pd.read_csv("gitignore/prediction_clip.csv")
prediction_clip_step = pd.read_csv("gitignore/prediction_clip_step.csv")

abs_score_clip_1 = 0
abs_score_clip_step_1 = 0
abs_score_clip_2 = 0
abs_score_clip_step_2 = 0

correct_labels_1 = df.label_niv1
clip_labels_1 = prediction_clip.label_niv1
clip_step_labels_1 = prediction_clip_step.label_niv1
correct_labels_2 = df.label_niv2
clip_labels_2 = prediction_clip.label_niv2
clip_step_labels_2 = prediction_clip_step.label_niv2

for i in range(len(df)):
    if clip_labels_1[i]==correct_labels_1[i]:
        abs_score_clip_1+=1
    if clip_step_labels_1[i]==correct_labels_1[i]:
        abs_score_clip_step_1+=1
    if clip_labels_2[i]==correct_labels_2[i]:
        abs_score_clip_2+=1
    if clip_step_labels_2[i]==correct_labels_2[i]:
        abs_score_clip_step_2+=1

score_clip_percent_niv1 = abs_score_clip_1/len(df)
score_clip_step_percent_niv1 = abs_score_clip_step_1/len(df)
score_clip_percent_niv2 = abs_score_clip_2/len(df)
score_clip_step_percent_niv2 = abs_score_clip_step_2/len(df)

print("CLIP directement au niveau 2 a trouvé {} pourcent des labels de niveau 1".format(score_clip_percent_niv1*100))
print("CLIP en mode step by step a trouvé {} pourcent des labels de niveau 1".format(score_clip_step_percent_niv1*100))
print("CLIP directement au niveau 2 a trouvé {} pourcent des labels de niveau 2".format(score_clip_percent_niv2*100))
print("CLIP en mode step by step a trouvé {} pourcent des labels de niveau 2".format(score_clip_step_percent_niv2*100))