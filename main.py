from utils import *
import numpy as np
import pickle
import cv2
from models.LogisticRegressioner import MyLogisticRegression, SKLearnLogisticRegression
from models.SVM import mySVM

if __name__ == '__main__':
    # Getting the data
    X_train, y_train = getData()
    X_test, y_test = getData(type='test')
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    shuffle(X_train, y_train)

    # Logistic Regression part
    myRegressioner = MyLogisticRegression()
    print("My Regressioner doing Logisitic Regression")
    myRegressioner.fit(X_train, y_train)
    print("SK-Learn doing Logisitic Regression")
    sklearnRegressioner = SKLearnLogisticRegression(1, 100, X_train, y_train)

    skPred = sklearnRegressioner.predict(X_test)
    myPred = myRegressioner.predict(X_test)
    skAccuracy = np.count_nonzero(skPred.reshape(skPred.shape[0], 1) == y_test) / y_test.shape[0]
    myAccuracy = np.count_nonzero(myPred == y_test) / y_test.shape[0]
    print("SK-Learn Accuracy on test data: ", skAccuracy, ", My Accuracy: ", myAccuracy)
