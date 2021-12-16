import sys, os
import pandas as pd
import torch
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy, Exact_Match_Ratio
import logging
from pathlib import Path
import shutil
from config import CFG


labels = pd.read_csv(CFG.path_label, index_col=False)
nb_labels = len(labels)


def preprocessing(labels, i):
    df_train = pd.read_csv(os.path.join(CFG.path_bert, 'Data/train.csv'), index_col=False)
    df_test = pd.read_csv(os.path.join(CFG.path_bert, 'Data/test.csv'), index_col=False)
    df_val = pd.read_csv(os.path.join(CFG.path_bert, 'Data/val.csv'), index_col=False)
    label_train=[]
    label_test=[]
    label_val=[]
    true_label = labels.fr.iloc[i]
    label = [true_label, 'Autre']
    df_label = pd.DataFrame(label, columns=['label'])
    df_label.to_csv(os.path.join(CFG.path_bert, 'Data', 'labels_one_vs_rest.csv'), index=False, header=False)

    for j in range(len(df_train)):
        if list(df_train.iloc[j].values()).index(1.0)==true_label:
            label_train.append(true_label)
        else:
            label_train.append('Autre')

    for j in range(len(df_test)):
        if list(df_test.iloc[j].values()).index(1.0)==true_label:
            label_test.append(true_label)
        else:
            label_test.append('Autre')

    for j in range(len(df_val)):
        if list(df_val.iloc[j].values()).index(1.0)==true_label:
            label_val.append(true_label)
        else:
            label_val.append('Autre')

    df_train_one_vs_rest = pd.DataFrame(list(zip(df_train.text.tolist(), label_train)), columns=['text', 'label'])
    df_train_one_vs_rest.to_csv(os.path.join(CFG.path_bert, 'Data','train_one_vs_rest.csv'), index=False)

    df_test_one_vs_rest = pd.DataFrame(list(zip(df_test.text.tolist(), label_test)), columns=['text', 'label'])
    df_test_one_vs_rest.to_csv(os.path.join(CFG.path_bert, 'Data', 'est_one_vs_rest.csv'), index=False)

    df_val_one_vs_rest = pd.DataFrame(list(zip(df_val.text.tolist(), label_val)), columns=['text', 'label'])
    df_val_one_vs_rest.to_csv(os.path.join(CFG.path_bert, 'Data', 'val_one_vs_rest.csv'), index=False)



def main_training_classifier(epochs, nb_labels=nb_labels, labels=labels):
    version_fine_tuned = len(os.listdir(os.path.join(CFG.path_models, 'CamemBERT_fine_tuned')))
    version_camembert = len(os.listdir(os.path.join(CFG.path_models, 'CamemBERT_one_vs_rest'))) + 1
    os.mkdir(
        os.path.join(CFG.path_models, 'CamemBERT_one_vs_rest', 'CamemBERT_one_vs_rest_v{}'.format(version_camembert)))
    os.mkdir(os.path.join(CFG.path, 'Tensorboard', 'CamemBERT_one_vs_rest',
                          'CamemBERT_one_vs_rest_v{}'.format(version_camembert)))
    MODEL_PATH = Path(
        os.path.join(CFG.path_models, 'CamemBERT_fine_tuned', 'CamemBERT_fine_tuned_v{}'.format(version_fine_tuned)))
    WGTS_PATH = Path(CFG.path_models, 'CamemBERT_fine_tuned/pytorch_model.bin')
    for i in range(nb_labels):
        preprocessing(labels,i)
        version_camembert = len(os.listdir(os.path.join(CFG.path_models, 'CamemBERT_one_vs_rest'))) + 1
        os.mkdir(os.path.join(CFG.path_models, 'CamemBERT_one_vs_rest',
                              'CamemBERT_one_vs_rest_v{}'.format(version_camembert), 'classifier_{}'.format(i)))
        os.mkdir(os.path.join(CFG.path, 'Tensorboard', 'CamemBERT_one_vs_rest',
                              'CamemBERT_one_vs_rest_v{}'.format(version_camembert), 'classifier_{}'.format(i)))
        BERT_PATH = Path(os.path.join(CFG.path_models, 'CamemBERT_one_vs_rest', 'CamemBERT_one_vs_rest_v{}'.format(version_camembert), 'classifier_{}'.format(i)))


        OUTPUT_DIR_BERT = Path(
            os.path.join(CFG.path, 'Tensorboard', 'CamemBERT_one_vs_rest', 'CamemBERT_one_vs_rest_v{}'.format(version_camembert), 'classifier_{}'.format(i)))
        databunch = BertDataBunch(DATA_PATH, DATA_PATH,
                                  tokenizer='camembert-base',
                                  train_file='train_one_vs_rest.csv',
                                  val_file='val_one_vs_rest.csv',
                                  label_file='labels_one_vs_rest.csv',
                                  text_col='text',
                                  label_col='label',
                                  batch_size_per_gpu=4,
                                  max_seq_length=512,
                                  multi_gpu=CFG.multi,
                                  multi_label=False,
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
            output_dir=OUTPUT_DIR_BERT,
            finetuned_wgts_path=WGTS_PATH,
            warmup_steps=500,
            multi_gpu=CFG.multi,
            is_fp16=True,
            multi_label=False,
            logging_steps=50)
        files = os.listdir(os.path.join(CFG.path_bert, 'Data/'))
        if 'cache' in files:
            shutil.rmtree(os.path.join(CFG.path_bert, 'Data', 'cache'))
        learner.fit(epochs=epochs,
                    lr=9e-5,
                    validate=True,  # Evaluate the model after each epoch
                    schedule_type="warmup_cosine",
                    optimizer_type="adamw")
        learner.save_model(BERT_PATH)


