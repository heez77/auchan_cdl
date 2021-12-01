import pandas as pd
import os

df = pd.read_csv("gitignore/df.csv")

# Pour Clip directement au niveau 2
prediction_clip = df
prediction_clip = prediction_clip[['image_name_MEDIASTEP','image_name','label']]
labels = [] # couple est une liste des correct_label dans l'ordre
for image in prediction_clip.image_name:
    index = (prediction_clip[prediction_clip.image_name==image].index.values)[0]
    # Trouver le folder où est image
    for folder in os.listdir("gitignore/RESULTS DATASET V2/RESULTS DATASET V2"):
        if image in os.path.join("gitignore/RESULTS DATASET V2/RESULTS DATASET V2", folder):
            prediction_clip.label[index]=folder
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
        if image in os.path.join("gitignore/RESULTS DATASET V2 STEP/RESULTS DATASET V2 STEP", folder):
            prediction_clip_step.label[index]=folder
            print('image n°{} done'.format(index))
prediction_clip_step.to_csv("gitignore/prediction_clip_step.csv")

# Comparer les deux colonnes label des deux nouveaux dataset avec la colonne label de df
abs_score_clip = 0
abs_score_clip_step = 0

correct_labels = df.label
clip_labels = prediction_clip.label
clip_step_labels = prediction_clip_step.label

for i in range(len(df)):
    if clip_labels[i]==correct_labels[i]:
        abs_score_clip+=1
    if clip_step_labels[i]==correct_labels[i]:
        abs_score_clip_step+=1

score_clip_percent = abs_score_clip/len(df)
score_clip_step_percent = abs_score_clip_step/len(df)

print("CLIP directement au niveau 2 a trouvé {} pourcent des labels".format(score_clip_percent))
print("CLIP en mode step by step a trouvé {} pourcent des labels".format(score_clip_step_percent))