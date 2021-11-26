import albumentations as A
import os
import numpy as np
from PIL import Image
import cv2

image = np.array(Image.open(os.path.join(os.getcwd(), "clip_archive", "picard.jpg")))
bbox = os.path.join(os.getcwd(), "clip_archive", "picard.csv")

transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.2),
        A.RandomRotate90(p=.3)
    ]
)

data = {'image': image, 'mask': bbox}

augmented = transform(data)

image, mask = augmented['image'], augmented['mask']

# transformed, mask = transform(image=image)["image"], transform(image=image)["mask"]



import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()