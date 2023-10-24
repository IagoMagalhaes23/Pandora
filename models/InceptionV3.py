'''
    Autores: Iago Magalhães e Vanessa Carvalho
    Data: 22/10/2023
    Descrição:
        - Arquiterura rede InceptionV3
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam

def inceptionV3():
    '''
        Função com arquitetura da rede CNN InceptionV3
        :return: modelo da rede CNN InceptionV3
    '''
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(units=5, activation='softmax'))

    optimizer = Adam(lr=0.0001, decay=1e-5)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    return model