import numpy as np
import json
import os
import cv2


with open ("open.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dirName = os.listdir("dataset")
dirName = [i for i in dirName if i.endswith(".png")]

# abcdefghijklmnopqrstu
imgArr = []
ladelArr = []

for i in dirName:
    img = cv2.imread(f"dataset/{i}.png")
    img = cv2.resize(img, (480, 640))

    img = img[120:-120, :, :]

    imgArr.append(img)
    ladelArr.append(data[i])

imgArr = np.array(imgArr)
ladelArr = np.array(ladelArr)

import tensorflow as tf
from keras import layers
import keras
InputL = layers.Input((240,640, 3))
x = layers.Conv2D(16, (3,3), activation="relu")(InputL)
x = layers.Conv2D(32, (3,3), activation="relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dense(24, activation="relu")(x)
x = layers.Dense(10, activation="relu")(x)
x = layers.Dense(5, activation="relu")(x)

model = keras.models.Model(inputs = [InputL], outputs = [x])
model.summary()