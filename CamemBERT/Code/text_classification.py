from os.path import dirname, abspath
import sys
root_path = dirname(abspath('text_classification.py'))
sys.path.append(root_path)
import nltk
import pandas as pd
import numpy as np
import re, os
from get_labels_from_directories import get_labels
import torch
from sklearn.model_selection import train_test_split
import tensorflow as tf
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
import unidecode
from config import CFG
from pathlib import Path

final_stopwords_list = nltk.corpus.stopwords.words('english') + nltk.corpus.stopwords.words('french')



REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
NEW_LINE = re.compile('\n')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(final_stopwords_list)

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()# lowercase text
    text = unidecode.unidecode(text)
    text = NEW_LINE.sub(' ',text) # replace NEW_LINE symbols in our texts by space
    text = REPLACE_BY_SPACE_RE.sub(' ',text)# replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('',text)# delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text


def preprocessing(df, texte):
    df_labels = get_labels()
    data = df.set_index('image').join(df_labels.set_index('image'))
    data.dropna(inplace=True)
    labels = data.label.unique().tolist()
    if texte == 'fournisseur':
        X = data['description_fournisseur'].tolist()
    else:
        X = data['description_produit'].tolist()

    dico = {label:i for i, label in enumerate(labels)}

    Y = data.label.tolist()
    Y = [dico[label] for label in Y]
    Y = tf.keras.utils.to_categorical(Y, num_classes=len(dico))
    Y = np.array(Y)
    X = [text_prepare(text) for text in X]
    X = np.array(X)

    x, x_test, y, y_test = train_test_split(X,Y,test_size=0.2,train_size=0.8)
    x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.25,train_size =0.75)

    df_lab = pd.DataFrame(columns=['label'])
    df_lab.label = labels

    df_train = pd.DataFrame(columns = ['text']+labels)
    df_val = pd.DataFrame(columns = ['text']+labels)
    df_test = pd.DataFrame(columns = ['text']+labels)

    df_train['text'] = x_train
    df_train[labels] = y_train

    df_val['text'] = x_val
    df_val[labels] = y_val

    df_test['text'] = x_test
    df_test[labels] = y_test

    df_train.to_csv(os.path.join(CFG.path_bert, 'Data/train.csv'))
    df_val.to_csv(os.path.join(CFG.path_bert,'Data/val.csv'))
    df_test.to_csv(os.path.join(CFG.path_bert,'Data/test.csv'))
    df_lab.to_csv(os.path.join(CFG.path_bert,'Data/labels.csv'), header=False, index=False)

def model():
    DATA_PATH = os.path.join(CFG.path_bert,'Data/')
    OUTPUT_DIR = os.path.join(CFG.path_bert,'Results/')
    labels = pd.read_csv(os.path.join(DATA_PATH,'labels.csv'), header=None, index_col=False)[0].tolist()
    databunch = BertDataBunch(DATA_PATH, DATA_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='train.csv',   
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col=labels,
                          batch_size_per_gpu=16,
                          max_seq_length=256,
                          multi_gpu=False,
                          multi_label=True,
                          model_type='bert')
    logger = logging.getLogger()
    device_cuda = torch.device("cuda")
    metrics = [{'name': 'accuracy', 'function': accuracy}]
    learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_DIR,
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=False,
						is_fp16=True,
						multi_label=True,
						logging_steps=50)

    #learner.lr_find(start_lr=1e-5,optimizer_type='lamb')

    return learner




def main():
    df = pd.read_csv(os.path.join(CFG.path_bert,'Data/text_detector_data.csv'))
    preprocessing(df, 'fournisseur')
    learner = model()
    learner.fit(epochs=1,
			lr=9e-1,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="lamb")
    
    learner.save_model(Path(CFG.path_models,'model_BERT'))

if __name__=='__main__':
    main()