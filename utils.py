import numpy as np
import pickle
import cv2

def shuffle(X,Y):
    """
    Shuffle the given data X & Y

    """
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

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
