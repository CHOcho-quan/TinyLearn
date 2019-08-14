import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt

def splitData(X, y, rate):
    """
    Split the data by the rate to validation set & training set

    """
    X_train = X[int(X.shape[0]*rate):]
    Y_train = y[int(y.shape[0]*rate):]
    x_val = X[:int(X.shape[0]*rate)]
    y_val = y[:int(y.shape[0]*rate)]
    return X_train, Y_train, x_val, y_val

def shuffle(X,Y):
    """
    Shuffle the given data X & Y

    """
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

def dataNormalize(X):
    X -= np.mean(X)
    X /= np.std(X)
    return X

def getDataCNN():
    """
    Getting  training or testing data for CNN

    """
    with open('./data/test.pickle', 'rb') as f:
        _, y_test = pickle.load(f)
    with open('./data/train.pickle', 'rb') as f:
        _, y_train = pickle.load(f)
    with open('./data/name.pickle', 'rb') as f:
        train_name, test_name = pickle.load(f)

    X_train = []
    X_test = []
    for i in train_name:
        img = cv2.imread(i)
        X_train.append(img)

    for i in test_name:
        img = cv2.imread(i)
        X_test.append(img)

    y_train = np.hstack((y_train, 1-y_train))
    y_test = np.hstack((y_test, 1-y_test))

    return np.array(X_train), np.array(X_test), y_train, y_test

def getData(type='train'):
    """
    Getting training or testing data from the stored pickle

    """
    if (type == 'test'):
        with open('./data/test.pickle', 'rb') as f:
            X_test, y_test = pickle.load(f)
            return X_test, y_test

    with open('./data/train.pickle', 'rb') as f:
        X_train, y_train = pickle.load(f)
    shuffle(X_train, y_train)
    return X_train, y_train
