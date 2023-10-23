'''
    Autores: Iago Magalhães e Vanessa Carvalho
    Data: 18/10/2023
    Descrição:
        - Implementa funções para leitura das imagens
        - Aplicação de filtro de realce e composição do dataset para treino, teste e validação
        - Plotagem dos gráficos de treinamento
'''

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readFiles(caminhos):
    '''
        Função para ler todos os arquivos de imagem em uma pasta
        :param caminhos: caminho dos arquivos de imagem
        : return: retorna uma lista com o endereço e nome da imagem e a respectiva classe
    '''
    cont = 0
    data_list = []

    for caminho, _, arquivo in os.walk(caminhos):
        cam = str(caminho.replace("\\", "/"))+"/"
        for file in arquivo:
            # print(file[-5])
            data_list.append([os.path.join(cam, file), file[-5]])

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

def plot_hist(hist):
    '''
        Função para plotar o gráfico de treinamento
        :param hist: recebe o histórico de treinamento da rede
    '''
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()