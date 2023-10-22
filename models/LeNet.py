'''
    Autores: Iago Magalhães e Vanessa Carvalho
    Data: 19/10/2023
    Descrição:
        - 
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam

def leNet():
    '''
    
    '''
    model = Sequential()

    model.add(Conv2D(filters=6, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(224,224,1)))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))

    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=2, activation = 'softmax'))

    optimizer = Adam(lr=0.0001, decay=1e-5)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    return model