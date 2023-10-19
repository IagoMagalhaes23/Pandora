'''
    Autores: Iago Magalhães e Vanessa Carvalho
    Data: 18/10/2023
    Descrição:
        - 
'''

import os
import cv2

from utils import dataset

caminhos = 'dataset/original/'

for caminho, _, arquivo in os.walk(caminhos):
    caminho = str(caminho.replace("\\", "/"))
    positions = []
    cont = 0
    if(caminho == 'dataset/original/valid'):
        for file in arquivo:
            if(file[-3:-1] == 'xm'):
                positions = dataset.readXML(caminho + '/' + file)
                image = cv2.imread(os.path.join(caminho + '/' + file[:-3] + 'jpg'))
                dataset.cropImage(image, positions, 'dataset/recortado/valid/', cont)
                cont += 1
    if(caminho == 'dataset/original/test'):
        for file in arquivo:
            if(file[-3:-1] == 'xm'):
                positions = dataset.readXML(caminho + '/' + file)
                image = cv2.imread(os.path.join(caminho + '/' + file[:-3] + 'jpg'))
                dataset.cropImage(image, positions, 'dataset/recortado/test/', cont)
                cont += 1
    if(caminho == 'dataset/original/train'):
        for file in arquivo:
            if(file[-3:-1] == 'xm'):
                positions = dataset.readXML(caminho + '/' + file)
                image = cv2.imread(os.path.join(caminho + '/' + file[:-3] + 'jpg'))
                dataset.cropImage(image, positions, 'dataset/recortado/train/', cont)
                cont += 1