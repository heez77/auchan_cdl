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
from tensorboardX import SummaryWriter
from fastprogress.fastprogress import master_bar, progress_bar
#######################################################################################################################
#                               Script des fonctions d'entrainement en OnevsRest                                      #
#                                                                                                                     #
#                 En mode One vs Rest, on créer autant de classifieur qu'il y a de labels et                          #
#               Chaque classifieur apprend à classifier un produit entre son label ou le reste.                       #
#######################################################################################################################

#---------------------------------------------------------------------------------------------------------------------#
# Modification de la classe proposée par fast-bert en rajoutant le renvoie du scheduler après un fit pour faire une sauvegarde manuelle du modèle 

class Learner(BertLearner):
    def __init__():
        super().__init__()
    def fit(
        self,
        epochs,
        lr,
        validate=True,
        return_results=False,
        schedule_type="warmup_cosine",
        optimizer_type="lamb",
    ):
        results_val = []
        tensorboard_dir = self.output_dir / "tensorboard"
        tensorboard_dir.mkdir(exist_ok=True)

        # Train the model
        tb_writer = SummaryWriter(tensorboard_dir)

        train_dataloader = self.data.train_dl
        if self.max_steps > 0:
            t_total = self.max_steps
            self.epochs = (
                self.max_steps // len(train_dataloader) // self.grad_accumulation_steps
                + 1
            )
        else:
            t_total = len(train_dataloader) // self.grad_accumulation_steps * epochs

        # Prepare optimiser
        optimizer = self.get_optimizer(lr, optimizer_type=optimizer_type)

        # get the base model if its already wrapped around DataParallel
        if hasattr(self.model, "module"):
            self.model = self.model.module

        # Get scheduler
        scheduler = self.get_scheduler(
            optimizer, t_total=t_total, schedule_type=schedule_type
        )

        # Parallelize the model architecture
        if self.multi_gpu is True:
            self.model = torch.nn.DataParallel(self.model)

        # Start Training
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataloader.dataset))
        self.logger.info("  Num Epochs = %d", epochs)
        self.logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
            self.data.train_batch_size * self.grad_accumulation_steps,
        )
        self.logger.info(
            "  Gradient Accumulation steps = %d", self.grad_accumulation_steps
        )
        self.logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        epoch_step = 0
        tr_loss, logging_loss, epoch_loss = 0.0, 0.0, 0.0
        self.model.zero_grad()
        pbar = master_bar(range(epochs))

        for epoch in pbar:
            epoch_step = 0
            epoch_loss = 0.0
            for step, batch in enumerate(progress_bar(train_dataloader, parent=pbar)):
                # Run training step and get loss
                loss = self.training_step(batch)

                tr_loss += loss.item()
                epoch_loss += loss.item()
                if (step + 1) % self.grad_accumulation_steps == 0:
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    if self.is_fp16:
                        # AMP: gradients need unscaling
                        self.scaler.unscale_(optimizer)

                    if self.is_fp16:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()

                    self.model.zero_grad()
                    global_step += 1
                    epoch_step += 1

                    if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                        if validate:
                            # evaluate model
                            results = self.validate()
                            for key, value in results.items():
                                tb_writer.add_scalar(
                                    "eval_{}".format(key), value, global_step
                                )
                                self.logger.info(
                                    "eval_{} after step {}: {}: ".format(
                                        key, global_step, value
                                    )
                                )

                        # Log metrics
                        self.logger.info(
                            "lr after step {}: {}".format(
                                global_step, scheduler.get_lr()[0]
                            )
                        )
                        self.logger.info(
                            "train_loss after step {}: {}".format(
                                global_step,
                                (tr_loss - logging_loss) / self.logging_steps,
                            )
                        )
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / self.logging_steps,
                            global_step,
                        )

                        logging_loss = tr_loss

            # Evaluate the model against validation set after every epoch
            if validate:
                results = self.validate()
                for key, value in results.items():
                    self.logger.info(
                        "eval_{} after epoch {}: {}: ".format(key, (epoch + 1), value)
                    )
                results_val.append(results)

            # Log metrics
            self.logger.info(
                "lr after epoch {}: {}".format((epoch + 1), scheduler.get_lr()[0])
            )
            self.logger.info(
                "train_loss after epoch {}: {}".format(
                    (epoch + 1), epoch_loss / epoch_step
                )
            )
            self.logger.info("\n")

        tb_writer.close()

        if return_results:
            return global_step, tr_loss / global_step, results_val, scheduler #Modification ici
        else:
            return global_step, tr_loss / global_step, scheduler # Modification ici 



#---------------------------------------------------------------------------------------------------------------------#


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
    DATA_PATH = Path(os.path.join(CFG.path_bert,'Data/'))
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


