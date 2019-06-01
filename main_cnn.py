import os, pickle
import numpy as np
import keras
from utils import *
from models.myCNN import MyConvNet
from models.CNNSolver import Solver
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = getDataCNN()
    _, y_train = getData()
    _, y_test = getData(type='test')
    N, L, H, C = X_train.shape
    X_train = X_train.reshape(N, C, H, -1)
    N, L, H, C = X_test.shape
    X_test = X_test.reshape(N, C, H, -1)
    print(X_train.shape)
    X_train, y_train = shuffle(X_train, y_train)

    model = MyConvNet()
    solver = Solver(model=model, data={'X_train':X_train, 'y_train':y_train, 'X_val':X_test, 'y_val':y_test})
    solver.train()

    # Getting the data
    X_train, X_test, y_train, y_test = getDataCNN()
    # print(y_test)
    X_train, y_train = shuffle(X_train, y_train)
    X_train, y_train, X_val, y_val = splitData(X_train, y_train, 0.3)
    # print(y_train.shape)

    # b x 3 x 96 x 96
    model = Sequential()
    # b x 32 x 46 x 46
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    model.add(BatchNormalization())
    # b x 32 x 21 x 21
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    callback = EarlyStopping(monitor="loss", patience=0.01, verbose=1, mode="auto")
    model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val), callbacks=[callback])

    kerasPred = model.predict(X_test)
    kerasPred = np.argmax(kerasPred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    # kerasPred[kerasPred >= 0.5] = 1
    # kerasPred[kerasPred < 0.5] = 0
    kerasAcc = np.count_nonzero(kerasPred == y_test) / y_test.shape[0]
    print("Keras accuracy is", kerasAcc)
