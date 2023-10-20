'''
    Autores: Iago Magalhães e Vanessa Carvalho
    Data: 18/10/2023
    Descrição:
        - Implementa funções para leitura das imagens, aplicação de filtro de realce e composição do dataset para treino, teste e validação
'''

import os
import cv2
import pandas as pd
import numpy as np

def readFiles(caminhos):
    '''
        Função para ler todos os arquivos de imagem em uma pasta
        :param caminhos: caminho dos arquivos de imagem
        : return: retorna uma lista com o endereço e nome da imagem
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
        Função para redimensionamento da imagens em 224x224 pixels, conversão para tons de cinza e aplicação do filtro Gaussiano
        :param cam: enderenço, nome e formato da imagem a ser tratada
        :return: retorna uma imagem redimensionada em tons de cinza
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
        Função para compor o dataset de treino, teste e validação
        :param df: recebe um dataframe com o endereço da imagem e seu label
        :return: retorna dois np.arrays com a imagem e o label
    '''
    data = []
    labels = []

    for img_path, label in df.values:
        data.append(filtro(img_path))
        labels.append(label)

    return np.array(data), np.array(labels)