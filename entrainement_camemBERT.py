from Entrainement.CamemBERT.camemBERT import main_training_BERT
import os
from config import CFG
import argparse

#######################################################################################################################
#                     Script d'execution pour la prédiction de la classification des produits.                        #
#                                                                                                                     #
#                            Ligne de commande : python3 entrainement_camemBERT.py                                    #
#             Possibilité d'ajouter l'argument --tuning pour faire du fine tuning sur le modèle camemBERT             #
#   Exemple si oui :                                                                                                  #
#                                python3 prediction_classification.py --tuning True                                   #
#######################################################################################################################


#         Argument --tuning pour la ligne de commande

parser = argparse.ArgumentParser(description='Fine Tuning')
parser.add_argument('--tuning', type=bool,
                    help='a boolean for the fine-tuning training of CamemBERT')
tuning = vars(parser.parse_args())['tuning']
if tuning==None:
    tuning = False

#---------------------------------------------------------------------------------------------------------------------#

def main():
    try:
        main_training_BERT(tuning)
    except Exception as e:
        if len(os.listdir(os.path.join(CFG.path, 'Data', 'Entrainement_camemBERT'))) == 0:
            print("Aucun fichier d'entrainement")
        else:
            print(e)


if __name__=='__main__':
    main()