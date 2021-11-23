#"C:\\Users\\geyma\\Documents\\Centrale Digital Lab\\Projet Auchan\\auchan_cdl\\gitignore\\données\\photos\\MEDIASTEP9657460_0.jpg"
import sys
import cv2
import pyocr
import numpy as np
from PIL import Image
import os

os.chdir("C:\\Users\\geyma\\Documents\\Centrale Digital Lab\\Projet Auchan\\auchan_cdl\\gitignore\\données\\photos")
image = "MEDIASTEP57466171_.jpg"
name = "MEDIASTEP57466171_"

#original
img = cv2.imread(image)
cv2.imwrite(f"1_{name}_original.png ",img)

#gray
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(f"2_{name}_gray.png ",img)

#threshold
th = 140
img = cv2.threshold(
    img
    , th
    , 255
    , cv2.THRESH_BINARY
)[1]
cv2.imwrite(f"3_{name}_threshold_{th}.png ",img)

#bitwise
img = cv2.bitwise_not(img)
cv2.imwrite(f"4_{name}_bitwise.png ",img)

tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
tool = tools[0]
print(tool)
res_fra = tool.image_to_string(
    Image.open(image)
    ,lang="fra")
res_eng = tool.image_to_string(
    Image.open(image)
    ,lang="eng")
res_test = tool.image_to_string(
    Image.open(image)
    ,lang="equ")
print("res_fra =", res_fra)
print("res_eng = ", res_eng)
print("res_test = ", res_test)
