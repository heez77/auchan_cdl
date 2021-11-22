from Dataset_Generator import DatasetAdaptor, EfficientDetDataModule
from EfficientDet_model import EfficientDetModel
import pandas as pd
from pytorch_lightning import Trainer
import torch
import numpy as np
import json, os


# ---------------------------- MAIN DEFAULT PARAMETERS ----------------------------#

dico = {'Logo AB': 1, 'Logo EU': 2, 'Bio': 3}
df_train = pd.read_csv('/home/jeremy/AUCHAN 2/pytorch_awesome/labels_train.csv')
df_val = pd.read_csv('/home/jeremy/AUCHAN 2/pytorch_awesome/labels_val.csv')
df_test = pd.read_csv('/home/jeremy/AUCHAN 2/pytorch_awesome/labels_test.csv')
image_path = '/home/jeremy/AUCHAN 2/IMAGES/BIO HD/images/'

# ---------------------------- MAIN FUNCTIONS ----------------------------#


def main():

    dataset_train = DatasetAdaptor(image_path + 'train/', df_train)
    dataset_val = DatasetAdaptor(image_path + 'val/', df_val)
    dataset_test = DatasetAdaptor(image_path + 'test/', df_test)
    dm = EfficientDetDataModule(dataset_train, dataset_val)
    model = EfficientDetModel(num_classes=3)  # Rajouter attribut img_size à 1200 ?
    trainer = Trainer(gpus=[0], max_epochs=100, num_sanity_val_steps=1)
    trainer.fit(model, dm)
    torch.save(model.state_dict(), '/home/jeremy/AUCHAN 2/pytorch_awesome/pytorch-awesome/modeles/efficientdet')
    model.eval()
    filenames = os.listdir(image_path + 'test/')
    n = len(filenames)
    images = []
    dico = dataset_test.dict_for_path()
    with open('/home/jeremy/AUCHAN 2/pytorch_awesome/pytorch-awesome/result/dico_img.json', 'w') as fp:
        json.dump(dico, fp)
    for i in range(n):
        image, _, _, _ = dataset_test.get_image_and_labels_by_idx(i)
        image.save('/home/jeremy/AUCHAN 2/pytorch_awesome/pytorch-awesome/result/image_{}.jpeg'.format(i))
        images.append(image)
    predicted_boxes, predicted_class_confidences, predicted_class_labels = model.predict(images)
    pb = np.array(predicted_boxes)
    pcc = np.array(predicted_class_confidences)
    pcl = np.array(predicted_class_labels)
    np.save('/home/jeremy/AUCHAN 2/pytorch_awesome/pytorch-awesome/result/pb.npy', pb)
    np.save('/home/jeremy/AUCHAN 2/pytorch_awesome/pytorch-awesome/result/pcc.npy', pcc)
    np.save('/home/jeremy/AUCHAN 2/pytorch_awesome/pytorch-awesome/result/pcl.npy', pcl)


if __name__ == '__main__':
    main()
