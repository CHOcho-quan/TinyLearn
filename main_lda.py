from utils import *
import numpy as np
import pickle
import cv2
from models.FisherModel import MyFisherModel, sklearnLDA

if __name__ == '__main__':
    # Getting the data
    X_train, y_train = getData()
    X_test, y_test = getData(type='test')
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_train, y_train = shuffle(X_train, y_train)

    myFisherModel = MyFisherModel()
    print("my Fisher Model doing regression")
    myFisherModel.fit(X_train, y_train)
    print("SK-Learn doing LDA")
    SKlearnLDA = sklearnLDA(X_train, y_train)

    myPred = myFisherModel.predict(X_test)
    skPred = SKlearnLDA.predict(X_test)
    skAccuracy = np.count_nonzero(skPred.reshape(skPred.shape[0], 1) == y_test) / y_test.shape[0]
    myAccuracy = np.count_nonzero(myPred.reshape(myPred.shape[0], 1) == y_test) / y_test.shape[0]
    print("SK-Learn Accuracy on test data: ", skAccuracy, " My Accuracy: ", myAccuracy)
