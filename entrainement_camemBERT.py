from Entrainement.CamemBERT.camemBERT import main_training_BERT
import os
from config import CFG



def main():
    try:
        main_training_BERT()
    except Exception as e:
        if len(os.listdir(os.path.join(CFG.path,'Data','Entrainement_camemBERT'))) ==0:
            print("Aucun fichier d'entrainement")
        else:
            print(e)


if __name__=='__main__':
    main()