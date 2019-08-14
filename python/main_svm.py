from utils import *
import numpy as np
import pickle
import cv2
from models.SVM import mySVM, sklearnSVM

if __name__ == '__main__':
    # Getting the data
    X_train, y_train = getData()
    X_test, y_test = getData(type='test')
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_train = dataNormalize(X_train)
    X_test = dataNormalize(X_test)

    with open('./data/name.pickle', 'rb') as f:
        train_name, test_name = pickle.load(f)
    # X_train, y_train = shuffle(X_train, y_train, train_name)


    # SVM Part
    # mySVMLinearRegressioner = mySVM()
    # mySVMRBFRegressioner = mySVM(kernel = 'rbf')
    # mySVMPolyRegressioner = mySVM(kernel = 'rbf')
    # print("My Regressioner doing SVM")
    # mySVMPolyRegressioner.fit(X_train, y_train)
    # mySVMRBFRegressioner.fit(X_train, y_train)
    # mySVMLinearRegressioner.fit(X_train, y_train)
    print("SK-Learn doing SVM")
    sklearnSVMRegressioner = sklearnSVM(X_train, y_train, kernel='rbf')
    print(len(sklearnSVMRegressioner.support_))
    for i in sklearnSVMRegressioner.support_:
        cv2.imshow("a", cv2.imread(train_name[i]))
        cv2.waitKey(0)

    # Predictions
    skPred = sklearnSVMRegressioner.predict(X_test)
    skAccuracy = np.count_nonzero(skPred.reshape(skPred.shape[0], 1) == y_test) / y_test.shape[0]
    # mySVMLinearPrediction = mySVMLinearRegressioner.predict(X_test)
    # mySVMRBFPrediction = mySVMRBFRegressioner.predict(X_test)
    # mySVMPolyPrediction = mySVMPolyRegressioner.predict(X_test)
    # myLinearAccuracy = np.count_nonzero(mySVMLinearPrediction == y_test) / y_test.shape[0]
    # myRBFAccuracy = np.count_nonzero(mySVMRBFPrediction.reshape(mySVMRBFPrediction.shape[0], 1) == y_test) / y_test.shape[0]
    # myPolyAccuracy = np.count_nonzero(mySVMPolyPrediction.reshape(mySVMPolyPrediction.shape[0], 1) == y_test) / y_test.shape[0]
    print("SK-Learn SVM Accuracy on test data: ", skAccuracy)#, "My SVM Accuracy: ", myPolyAccuracy)
