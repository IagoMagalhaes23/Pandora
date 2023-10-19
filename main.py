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

from utils import functions as fct

caminho_train = 'dataset/recortado/train'
caminho_test = 'dataset/recortado/test'
caminho_valid = 'dataset/recortado/valid'

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

print(df_train.shape)
print(df_valid.shape)
print(df_test.shape)

X_train, y_train = fct.compose_dataset(df_train)
X_test, y_test = fct.compose_dataset(df_test)
X_valid, y_valid = fct.compose_dataset(df_valid)

print('Treino shape: {}, Labels shape: {}'.format(X_train.shape, y_train.shape))
print('Valid shape: {}, Labels shape: {}'.format(X_valid.shape, y_valid.shape))
print('Teste shape: {}, Labels shape: {}'.format(X_test.shape, y_test.shape))

X_train, X_test, y_train, y_test, y_valid = fct.expandDataset(X_train, X_test, y_train, y_test, y_valid)

