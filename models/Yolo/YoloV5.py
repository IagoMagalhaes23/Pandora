'''
    Autores: Iago Magalhães e Vanessa Carvalho
    Data: 22/10/2023
    Descrição:
        - Implementação da rede YoloV5
'''

import os

def yoloV5():
    '''
        Função
    '''
    os.system('python models/Yolo/yoloV5/train.py --img 640 --batch 16 --epochs 5 --data data.yaml')