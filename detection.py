## Import librairies
from imageai.Detection.Custom import DetectionModelTrainer, CustomObjectDetection
import os
import cv2
import tensorflow as tf
import numpy as np



## Global Variables + GPU
current_path = os.getcwd()
#model_type = "retinanet"
dataset_directory = os.path.join(current_path, "dataset")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)



## Preprocess
import os
import cv2
import numpy as np
import shutil
import xml.etree.ElementTree as ET
for image in os.listdir ("dataset/train/images"):
   name, extension = os.path.splitext(image)
   img = cv2.imread("dataset/train/images/" + image)
   height = np.size(img, 0)
   width = np.size(img, 1)
   # Gray-scale images
   gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   if (("dataset/train/images/" + name + "_grey" + extension) not in os.listdir("dataset/train/images")) and ("grey" not in name):
      cv2.imwrite("dataset/train/images/" + name + "_grey" + extension, gray_image)
      doc = ET.parse("dataset/train/annotations/" + name + ".xml")
      root = doc.getroot()
      root[2].text = current_path + "\dataset\\train\\images\\" + name + "_grey.jpeg"
      doc.write("dataset/train/annotations/" + name + "_grey.xml")



for image in os.listdir ("dataset/train/images"):
   if ("grey" in image):
      os.remove("dataset/train/images/" + image)
for image in os.listdir ("dataset/train/annotations"):
   if ("grey" in image):
      os.remove("dataset/train/annotations/" + image)

## Train
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory = dataset_directory)
trainer.setTrainConfig(object_names_array = ["Logo AB", "Logo EU", "Logo BIO"], batch_size=1, num_experiments=100)  # We can train from pretrained model here
#train_from_pretrained_model="dataset/models/detection_model-ex-013--loss-0028.823.h5"
#trainer.trainModel()



## Evaluate
trainer.evaluateModel(model_path="dataset/models/detection_model-ex-013--loss-0028.823.h5", json_path="dataset/json/detection_config.json")



## Detect
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("dataset/models/detection_model-ex-037--loss-0009.851.h5")
detector.setJsonPath("dataset/json/detection_config.json")
detector.loadModel()
for img_test in os.listdir("dataset/test"):
   name, extension = os.path.splitext("dataset/test/" + img_test)
   detector.detectObjectsFromImage(input_image = name + extension, output_image_path = name + "_bbox" + extension)