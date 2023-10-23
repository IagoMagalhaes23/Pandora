'''
    Autores: Iago Magalhães e Vanessa Carvalho
    Data: 22/10/2023
    Descrição:
        - Implementação da rede YoloV5
'''

import os

def yoloV5():
    '''
        Função para executar o treinamento do Yolo V5
    '''
    os.system('python models/Yolo/yolov5/train.py --img 640 --batch-size 16 --epochs 5 --data models/Yolo/soc_mcq.v1i.yolov5pytorch/data.yaml')