#"C:\\Users\\geyma\\Documents\\Centrale Digital Lab\\Projet Auchan\\auchan_cdl\\gitignore\\données\\photos\\MEDIASTEP9657460_0.jpg"
import sys
import cv2
import pyocr
import numpy as np
from PIL import Image
import os
import pandas as pd

os.chdir("C:\\Users\\geyma\\Documents\\Centrale Digital Lab\\Projet Auchan\\auchan_cdl\\gitignore\\données\\photos")

def clean_res(res):
    # Permet de renvoyer une liste des mots trouvés par l'algorithme OCR
    # output : ['Mot1','Mot2',...]
    cleaned_res = []
    for character in res:
        if character.isalpha():
            cleaned_res.append(character)
        elif character == '\n' or character == ' ':
            cleaned_res.append('')
    for n in range (len(cleaned_res)-1):
        if cleaned_res[n].isalpha():
            while cleaned_res[n+1].isalpha():
                cleaned_res[n] += cleaned_res[n+1]
                del cleaned_res[n+1]
                cleaned_res.append('')
    element = 0
    while element != len(cleaned_res):
        if cleaned_res[element] == '':
            del cleaned_res[element]
        else :
            element += 1
    return(cleaned_res)

def levenshtein(chaine1, chaine2):
    taille_chaine1 = len(chaine1) + 1
    taille_chaine2 = len(chaine2) + 1
    levenshtein_matrix = np.zeros ((taille_chaine1, taille_chaine2))
    for x in range(taille_chaine1):
        levenshtein_matrix [x, 0] = x
    for y in range(taille_chaine2):
        levenshtein_matrix [0, y] = y

    for x in range(1, taille_chaine1):
        for y in range(1, taille_chaine2):
            if chaine1[x-1] == chaine2[y-1]:
                levenshtein_matrix [x,y] = min(
                    levenshtein_matrix[x-1, y] + 1,
                    levenshtein_matrix[x-1, y-1],
                    levenshtein_matrix[x, y-1] + 1
                )
            else:
                levenshtein_matrix [x,y] = min(
                    levenshtein_matrix[x-1,y] + 1,
                    levenshtein_matrix[x-1,y-1] + 1,
                    levenshtein_matrix[x,y-1] + 1
                )
    return (levenshtein_matrix[taille_chaine1 - 1, taille_chaine2 - 1])

def get_best_predictions(mot, dico):
    # Donne le(s) mot(s) du dico le(s) plus proche(s) du mot
    # inputs : 
    # - mot : 'Mot'
    # - dico : ['Mot1','Mot2',...]
    # Output : ['Motdudico']
    levenshtein_dist = []
    best_predictions=[]
    for word in dico :
        levenshtein_dist.append(levenshtein(mot,word))
    best = min(levenshtein_dist)
    for n in range (len(levenshtein_dist)):
        if levenshtein_dist[n] == best :
            best_predictions.append(dico[n])
    return(best_predictions)

def OCR(image):
    #image = "MEDIASTEP57466700_.jpg"
    name = os.path.basename(image)

    # #original
    # img = cv2.imread(image)
    # cv2.imwrite(f"1_{name}_original.png ",img)

    # #gray
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(f"2_{name}_gray.png ",img)

    # #threshold
    # th = 140
    # img = cv2.threshold(
    #     img
    #     , th
    #     , 255
    #     , cv2.THRESH_BINARY
    # )[1]
    # cv2.imwrite(f"3_{name}_threshold_{th}.png ",img)

    # #bitwise
    # img = cv2.bitwise_not(img)
    # cv2.imwrite(f"4_{name}_bitwise.png ",img)

    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    tool = tools[0]
    res_fra = tool.image_to_string(
        Image.open(image)
        ,lang="fra")
    # res_eng = tool.image_to_string(
    #     Image.open(f"3_{name}_threshold_{th}.png ")
    #     ,lang="eng")
    return(res_fra)

def write_df(all_prediction, df):


def main():
    fichier = open("C:\\Users\\geyma\\Documents\\Centrale Digital Lab\\Projet Auchan\\auchan_cdl\\gitignore\\liste_francais\\liste_francais.txt", "r")
    str = fichier.read()
    dico = list(str.split('\n'))
    df = pd.read_csv("C:\\Users\\geyma\\Documents\\Centrale Digital Lab\\Projet Auchan\\auchan_cdl\\gitignore\\données\\auchan_product_media_sample.csv")
    df = df.assign(words_in_the_image=None)
    for img in os.listdir():
        res_fra = OCR(img)
        all_predictions = []
        for word in clean_res(res_fra):
            all_predictions.append(get_best_predictions(word,dico))
        to_remove=[]
        for x in all_predictions:
            if len(x)>3:
                to_remove.append(x)
        for remove in to_remove:
            all_predictions.remove(remove)
        print(all_predictions)
        print(img)
        ind = df[df['photo_id']==img].index
        df.at[ind,'words_in_the_image'] = all_predictions
        df.head()
        

if __name__== '__main__':
    main()