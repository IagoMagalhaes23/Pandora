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

# import cv2 
# from ultralytics import YOLO

# im = cv2.imread('dataset/recortado/valid/0_A.png')
# modelo = YOLO("models/Yolo/best.pt")
# results = modelo.predict(im, conf=0.35)
# for result in results:
#   print(result.boxes.cls)