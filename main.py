from utils import *
import numpy as np
import pickle
import cv2
from models.LogisticRegressioner import MyLogisticRegression, SKLearnLogisticRegression
from models.SVM import mySVM, sklearnSVM

if __name__ == '__main__':
    # Getting the data
    X_train, y_train = getData()
    X_test, y_test = getData(type='test')
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    shuffle(X_train, y_train)

    # SVM Part
    mySVMLinearRegressioner = mySVM()
    mySVMRBFRegressioner = mySVM(kernel = 'rbf')
    print("My Regressioner doing SVM")
    mySVMRBFRegressioner.fit(X_train, y_train)
    # mySVMLinearRegressioner.fit(X_train, y_train)
    print("SK-Learn doing SVM")
    sklearnSVMRegressioner = sklearnSVM(X_train, y_train)

    # Predictions
    skPred = sklearnSVMRegressioner.predict(X_test)
    skAccuracy = np.count_nonzero(skPred.reshape(skPred.shape[0], 1) == y_test) / y_test.shape[0]
    # mySVMLinearPrediction = mySVMLinearRegressioner.predict(X_test)
    mySVMRBFPrediction = mySVMRBFRegressioner.predict(X_test)
    # myLinearAccuracy = np.count_nonzero(mySVMLinearPrediction == y_test) / y_test.shape[0]
    myRBFAccuracy = np.count_nonzero(mySVMRBFPrediction == y_test) / y_test.shape[0]
    print("SK-Learn SVM Accuracy on test data: ", skAccuracy, "My SVM Accuracy: ", myRBFAccuracy)

    # Logistic Regression part
    myLangevinRegressioner = MyLogisticRegression(optimization = 'langevin')
    mySGDRegressioner = MyLogisticRegression(optimization = 'sgd')
    print("My Regressioner doing Logisitic Regression")
    myLangevinRegressioner.fit(X_train, y_train)
    mySGDRegressioner.fit(X_train, y_train)
    print("SK-Learn doing Logisitic Regression")
    sklearnRegressioner = SKLearnLogisticRegression(1, 100, X_train, y_train)

    # Predictions
    skPred = sklearnRegressioner.predict(X_test)
    mySGDPred = mySGDRegressioner.predict(X_test)
    myLangevinPred = myLangevinRegressioner.predict(X_test)
    skAccuracy = np.count_nonzero(skPred.reshape(skPred.shape[0], 1) == y_test) / y_test.shape[0]
    mySGDAccuracy = np.count_nonzero(mySGDPred == y_test) / y_test.shape[0]
    myLangevinAccuracy = np.count_nonzero(myLangevinPred == y_test) / y_test.shape[0]
    print("SK-Learn Accuracy on test data: ", skAccuracy, ", My SGD Accuracy: ", mySGDAccuracy, "My langevin Accuracy: ", myLangevinAccuracy)
