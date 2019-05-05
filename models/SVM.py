import numpy as np
import cv2
import math
import pickle
from utils import *
from sklearn.svm import SVC

class mySVM:
    """
    My SVM Implementation

    """
    def __init__(self, regularization = 0.5, max_iter = 100, tolerance = 1e-3, kernel = 'linear'):
        """
        Initialization of the SVM classifier including 3 kinds of kernels

        """
        self.C = regularization
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.bias = None
        self.w = None
        self.alpha = None
        if (kernel == 'rbf'):
            self.innerProduct = self._rbfKernel
        elif (kernel == 'linear'):
            self.innerProduct = self._inner
        elif (kernel == 'cosine'):
            self.innerProduct = self._cosKernel
        else:
            raise Exception("No such kernel")

    def _inner(self, X, Y):
        """
        Calculating the inner product of two matrixes or vectors

        """
        return np.transpose(X).dot(Y)

    def _rbfKernel(self, X, Y):
        """
        RBF Kernel calculating with matrixes or vecotrs X and Y

        """
        if X.ndim == 1:
            return np.exp(-np.sum((X - Y)**2) / 2)
        return np.exp(-np.sum((X - Y)**2, axis=1) / 2)

    def _cosKernel(self, X, Y):
        """
        Cosine similarity Kernel calculating with matrixes or vectors X and Y

        """
        return np.transpose(X).dot(Y) / np.sqrt(np.sum(X**2))*np.sqrt(np.sum(Y**2))

    def fit(self, X_training, y_training):
        """
        Train the linear SVM with training data X_train & y_train
        X_train : Input training features with shape n x p
        y_train : Input label with the shape n x 1

        """
        pass

    def predict(self, X):
        """
        Make Predictions based on the learned w & bias

        """
        pred = X.dot(self.w) + self.bias
        pred[pred < 0] = 0
        pred[pred >= 0] = 1
        return pred

def sklearnSVM(X_train, y_train, C = 1.0, max_iter = 100):
    """
    SKLearn SVM to be tested with MySVM results
    C : The penalty term of SVM
    max_iter : Max hard iteration of the SVM

    """
    clf = SVC(C=C, max_iter=max_iter).fit(X_train, y_train)
    return clf
