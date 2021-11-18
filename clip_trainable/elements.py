from CFG import CFG
import os

def get_elements():
    list_directories = os.listdir(CFG.path)

    image_filenames = []
    captions = []

    for folder in list_directories:
        for file in os.listdir(os.path.join(CFG.path, folder)):
            image_filenames += [file]
            captions += [folder]
    return captions, image_filenames