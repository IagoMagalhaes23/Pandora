'''
    Autores: Iago Magalhães e Vanessa Carvalho
    Data: 22/10/2023
    Descrição:
        - Arquitetura de rede EfficientNetB0
'''

from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
import tensorflow as tf

def efficientnetB0():
    NUM_CLASSES = 6
    IMG_SIZE = 224
    size = (IMG_SIZE, IMG_SIZE)

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    # Using model without transfer learning

    outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(inputs)

    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"] )

    model.summary()

    return model