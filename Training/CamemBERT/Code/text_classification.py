from os.path import dirname, abspath
import sys, os
import nltk
import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
import tensorflow as tf
from fast_bert.data_cls import BertDataBunch
from fast_bert.data_lm import BertLMDataBunch
from fast_bert.learner_lm import BertLMLearner
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy, Exact_Match_Ratio
import logging
import unidecode
from pathlib import Path
root_path = dirname(abspath('text_classification.py'))
sys.path.append(root_path)
import shutil
from config import CFG
from Training.CamemBERT.Code.get_labels_from_directories import get_labels
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


def preprocessing(df,model = '1'):
    DATA_PATH = os.path.join(CFG.path_bert,'Data/')
    df_labels = get_labels(path='/home/jeremy/AUCHAN 2/IMAGES/Correct_labels')
    data = df.set_index('image').join(df_labels.set_index('image'))
    data.dropna(inplace=True)
    labels = data.label.unique().tolist()
    X1 = data['description_fournisseur'].tolist()
    X2 = data['description_produit'].tolist()
    X = [X1[i] +' '+ X2[i] for i in range(len(X1))]
    dico = {label:i for i, label in enumerate(labels)}
    imgs = data.index.tolist()
    Y = data.label.tolist()
    Y = [dico[label] for label in Y]
    Y = tf.keras.utils.to_categorical(Y, num_classes=len(dico))
    Y = np.array(Y)
    X = [text_prepare(text) for text in X]
    X = np.array(X)

    x, x_test, y, y_test, img, img_test = train_test_split(X,Y,imgs,test_size=0.2,train_size=0.8)
    x_train, x_val, y_train, y_val, img_train, img_val = train_test_split(x,y,img,test_size = 0.25,train_size =0.75)

    df_lab = pd.DataFrame(columns=['label'])
    df_lab.label = labels

    df_train = pd.DataFrame(columns = ['text']+labels)
    df_val = pd.DataFrame(columns = ['text']+labels)
    df_test = pd.DataFrame(columns = ['text']+labels)

    df_img_train = pd.DataFrame(columns=['image'])
    df_img_val = pd.DataFrame(columns=['image'])
    df_img_test = pd.DataFrame(columns=['image'])

    df_train['text'] = x_train
    df_train[labels] = y_train

    df_val['text'] = x_val
    df_val[labels] = y_val

    df_test['text'] = x_test
    df_test[labels] = y_test

    df_img_train.image = img_train
    df_img_test.image = img_test
    df_img_val.image = img_val
    df_img_train.to_csv(os.path.join(CFG.path_bert, 'Data/img_train.csv'))
    df_img_val.to_csv(os.path.join(CFG.path_bert, 'Data/img_val.csv'))
    df_img_test.to_csv(os.path.join(CFG.path_bert, 'Data/img_test.csv'))
    df_train.to_csv(os.path.join(CFG.path_bert, 'Data/train.csv'))
    df_val.to_csv(os.path.join(CFG.path_bert,'Data/val.csv'))
    df_test.to_csv(os.path.join(CFG.path_bert,'Data/test.csv'))
    df_lab.to_csv(os.path.join(CFG.path_bert,'Data/labels.csv'), header=False, index=False)
    if model=='2':
        return X
    
    
    
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
                          batch_size_per_gpu=4,
                          max_seq_length=512,
                          multi_gpu=False,
                          multi_label=True,
                          model_type='roberta')
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

def model2():
    DATA_PATH = Path(os.path.join(CFG.path_bert,'Data/'))
    MODEL_PATH = Path(CFG.path_models,'CamemBERT_fine_tuned/')
    WGTS_PATH = Path(CFG.path_models,'CamemBERT_fine_tuned/pytorch_model.bin')
    df = pd.read_csv(os.path.join(CFG.path_bert,'Data/text_detector_data.csv'))
    texts = preprocessing(df, model='2')
    logger = logging.getLogger()
    databunch_lm = BertLMDataBunch.from_raw_corpus(
					data_dir=DATA_PATH,
					text_list=texts,
					tokenizer='camembert-base',
					batch_size_per_gpu=4,
					max_seq_length=512,
                    multi_gpu=False,
                    model_type='camembert-base',
                    logger=logger)
    lm_learner = BertLMLearner.from_pretrained_model(
                            dataBunch=databunch_lm,
                            pretrained_path='camembert-base',
                            output_dir=MODEL_PATH,
                            metrics=[],
                            device=torch.device(CFG.device),
                            logger=logger,
                            multi_gpu=False,
                            logging_steps=50,
                            fp16_opt_level="O2")
    lm_learner.fit(epochs=30,
            lr=1e-4,
            validate=True,
            schedule_type="warmup_cosine",
            optimizer_type="adamw")
    lm_learner.validate()  
    lm_learner.save_model(MODEL_PATH)      
    OUTPUT_DIR = os.path.join(CFG.path_bert,'Results','CamemBERT')
    labels = pd.read_csv(os.path.join(DATA_PATH,'labels.csv'), header=None, index_col=False)[0].tolist()
    databunch = BertDataBunch(DATA_PATH, DATA_PATH,
                          tokenizer='camembert-base',
                          train_file='train.csv',   
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col=labels,
                          batch_size_per_gpu=4,
                          max_seq_length=512,
                          multi_gpu=False,
                          multi_label=True,
                          model_type='camembert-base')
    logger = logging.getLogger()
    device_cuda = torch.device("cuda")
    metrics = [{'name': 'Exact_Match_Ratio', 'function': Exact_Match_Ratio}]
    learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path=MODEL_PATH,
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_DIR,
						finetuned_wgts_path=WGTS_PATH,
						warmup_steps=500,
						multi_gpu=False,
						is_fp16=True,
						multi_label=True,
						logging_steps=50)
    return learner

def main():
    files=os.listdir(os.path.join(CFG.path_bert,'Data/'))
    if 'cache' in files:
        shutil.rmtree(os.path.join(CFG.path_bert,'Data','cache'))
    df = pd.read_csv(os.path.join(CFG.path_bert,'Data/text_detector_data.csv'))
    preprocessing(df)
    learner = model()
    learner.fit(epochs=20,
			lr=9e-5,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="lamb")
    learner.validate()  
    learner.save_model(Path(CFG.path_models,'model_BERT'))

def main2():
    files=os.listdir(os.path.join(CFG.path_bert,'Data/'))
    if 'cache' in files:
        shutil.rmtree(os.path.join(CFG.path_bert,'Data','cache'))
    learner = model2()
    learner.fit(epochs=150,
			lr=9e-5,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="adamw")
    learner.save_model(Path(CFG.path_models,'model_camemBERT'))
if __name__=='__main__':
    main2()