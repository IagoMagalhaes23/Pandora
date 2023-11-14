'''
    Autor: Iago e Vanessa
    Data: 13/11/2023
    Descrição:
        - 
'''

import os
import sys
import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, request, jsonify



path = os.path.abspath("../")
sys.path.append(path)

from utils import functions as fct

app = Flask(__name__)
imagem = ''

@app.route('/upload_imagem', methods=['POST'])
def upload_imagem():
    '''
        Rota para realizar upload de vídeo
    '''
    if 'imagem' not in request.files:
        return 'Nenhum arquivo de imagem enviado', 400

    imagem_file = request.files['imagem']

    if imagem_file.filename == '':
        return 'Nome de arquivo de imagem vazio', 400

    imagem_file.save('files/' + imagem_file.filename)
    endereco = str('files/' + imagem_file.filename)

    return jsonify({'endereco': endereco, 'mensagem': 'Imagem recebida com sucesso', 'status': 200})

@app.route('/predicao', methods=['GET'])
def predicao():
    '''
        Rota para realizar a predição da imagem
    '''
    imagem_path = request.args.get('imagem_path')
    modelo = load_model('../weights/efficientnetb0.hdf5')
    im = cv2.imread(imagem_path)
    im = cv2.resize(im, (224, 224))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(gray, ( 3, 3), 0)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    image = fct.prepararImagem(img)
    ret = modelo.predict(image, batch_size=1)
    resultado = fct.mostraCateg(ret)
    resposta = resultado
    return jsonify({'resultado': resultado}), 200


@app.route('/gabarito', methods=['GET'])
def gabarito():
    '''
        Rota para pegar o gabarito
    '''
    gabarito = request.args.get('resposta')
    
    return jsonify({'resultado': gabarito}), 200

@app.route('/resultado', methods=['GET'])
def resultado():
    '''
        Rota para pegar o gabarito
    '''
    gabarito = request.args.get('resposta')
    predicao = request.args.get('predicao')
    
    if(gabarito == predicao):
        return jsonify({'resultado': 'Acertou!'}), 200
    else:
        return jsonify({'resultado': 'Errou!'}), 200

if(__name__) == '__main__':
    app.run(debug=True)