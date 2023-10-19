'''
    Autores: Iago Magalhães e Vanessa Carvalho
    Data: 18/10/2023
    Descrição:
        - 
'''

import os
import cv2
import random
import pandas as pd
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

def readFiles(caminhos):
    '''
    
    '''
    cont = 0
    data_list = []

    for caminho, _, arquivo in os.walk(caminhos):
        cam = str(caminho.replace("\\", "/"))+"/"
        for file in arquivo:
            data_list.append([os.path.join(cam, file), 0])

    return data_list

def filtro(cam):
    '''
    
    '''
    img = cv2.imread(cam)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.0
    img = cv2.GaussianBlur(img, ( 3, 3), 0)
    img = np.reshape(img, (224,224,1))
    return img

def compose_dataset(df):
    '''
    
    '''
    data = []
    labels = []

    for img_path, label in df.values:
        data.append(filtro(img_path))
        labels.append(label)

    return np.array(data), np.array(labels)

def expandDataset(X_train, X_test, y_train, y_test, y_valid):
    '''
    
    '''
    X_train = datagen.fit(X_train)
    X_test = datagen.fit(X_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_valid = to_categorical(y_valid)

    return X_train, X_test, y_train, y_test, y_valid