'''
    Autores: Iago Magalhães e Vanessa Carvalho
    Data: 18/10/2023
    Descrição:
        - Ler arquivos XML e extrai as posições da ROI
        - Gerar novas imagens com as regiões recortadas
        - Organiza os arquivos em pastas para treino, teste e validação
'''

from bs4 import BeautifulSoup

import cv2
import os

def readXML(file):
    '''
        Função para ler arquivos XML e pegar os valores das posições da mucosa em x, y, w e h.
        :param file: recebe o endereço do arquivo
        :return: retorna uma lista com os de x, y, w e h
    '''
    with open(file, 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml")
    box = Bs_data.find_all('bndbox')

    results = []

    for name in box:
        results.append(name.text)
    
    return results

def cropImage(image, positions, caminho, name, classe):
    '''
        Função para recortar a mucosa da imagem original
        :param image: recebe a imagem original do dataset
        :param positions: recebe uma lista com as posições x, y, w e h
        :param caminho: recebe o endereço onde a imagem deve ser salva
        :param name: nome da imagem a ser salva
        :param classe: classe da imagem resultante
    '''
    string = ''.join(positions[0].splitlines())
    x = int(string[0:3])
    y = int(string[6:9])
    w = int(string[3:6])
    h = int(string[9:12])

    # print(string)
    # print(x, y, w, h)

    roi = image[y:h, x:w]

    try:
        cv2.imwrite(os.path.join(caminho, '%d.png') %name, roi)
    except:
        print('Erro ao salvar imagem: {}_{}.png'.format(name, classe))