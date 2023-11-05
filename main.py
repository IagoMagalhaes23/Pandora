'''
    Autores: Iago Magalhães e Vanessa Carvalho
    Data: 18/10/2023
    Descrição:
        - Arquivo principal
        - Ler imagens e as divide em treino, teste e validação
        - Executa o treinamento de todos os modelos de CNN, GNN e ViT
'''

import os
import cv2
import random
import pandas as pd
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models import AlexNet, EfficientNet, InceptionV3, LeNet, VanessinhaNet, ViT
# from models.Yolo import  YoloV5, YoloV8
from utils import functions as fct

caminho_train = 'dataset/recortado/train'
caminho_test = 'dataset/recortado/test'
caminho_valid = 'dataset/recortado/valid'

size = 35
filtro = 5

callback = EarlyStopping(monitor='loss', patience=5, mode='min')

data_train = fct.readFiles(caminho_train)
data_test= fct.readFiles(caminho_test)
data_valid = fct.readFiles(caminho_valid)

random.shuffle(data_train)
random.shuffle(data_test)
random.shuffle(data_valid)

df_train = pd.DataFrame(data_train, columns=['image', 'label'])
df_test = pd.DataFrame(data_test, columns=['image', 'label'])
df_valid = pd.DataFrame(data_valid, columns=['image', 'label'])

df_train = df_train.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)
df_valid = df_valid.reset_index(drop = True)

print('----------------------------- SHAPES DF -----------------------------')
print(df_train.shape)
print(df_valid.shape)
print(df_test.shape)
print('---------------------------------------------------------------------')

X_train, y_train = fct.compose_dataset(df_train, size, filtro)
X_test, y_test = fct.compose_dataset(df_test, size, filtro)
X_valid, y_valid = fct.compose_dataset(df_valid, size, filtro)

print('----------------------------- SHAPES LABEL -----------------------------')
print('Treino shape: {}, Labels shape: {}'.format(X_train.shape, y_train.shape))
print('Valid shape: {}, Labels shape: {}'.format(X_valid.shape, y_valid.shape))
print('Teste shape: {}, Labels shape: {}'.format(X_test.shape, y_test.shape))
print('------------------------------------------------------------------------')

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

datagen.fit(X_train)
datagen.fit(X_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_valid = to_categorical(y_valid)

# modelAlex = LeNet.leNet()
modelAlex = AlexNet.alexNet()
# modelAlex = InceptionV3.inceptionV3()
history = modelAlex.fit(X_train, y_train, batch_size=1, epochs = 10, verbose = 1, callbacks=[callback])
modelAlex.summary()
# modelAlex.save(filepath='modelalexnet.hdf5')




# #Effiecient net b0
# from sklearn.preprocessing import LabelEncoder , OneHotEncoder
# from sklearn.compose import ColumnTransformer

# ct = ColumnTransformer([('my_ohe', OneHotEncoder(), [0])], remainder='passthrough')
# y_train = ct.fit_transform(y_train) #.toarray()

# modelo = EfficientNet.efficientnetB0()
# # epochs = 10  # @param {type: "slider", min:10, max:100}
# hist = modelo.fit(X_train, y_train, epochs=30, verbose=2)
# hist = modelo.fit(X_train, epochs=epochs, validation_data=X_test, verbose=2)




#Yolo V5
# YoloV5.yoloV5()