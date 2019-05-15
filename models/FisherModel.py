import numpy as np
import cv2
import os
import math
import pickle
from utils import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class MyFisherModel:
    """
    A Fisher model to do face detection using LDA method

    """
    def __init__(self):
        self.beta = None
        self.threshold = 0

    def fit(self, X_train, y_train):
        """
        Fit the LDA model with training set X_train & label y_train

        """
        N, P = X_train.shape
        # First getting the \mu & \Sigma of the two classes
        negIndex = np.where(y_train.reshape(1, N)==0)
        posIndex = np.where(y_train.reshape(1, N)==1)
        mu0 = np.mean(X_train[negIndex[0], :], axis=0).reshape(1, P)
        mu1 = np.mean(X_train[posIndex[0], :], axis=0).reshape(1, P)

        sigma0 = np.zeros(shape=(P, P))
        sigma1 = np.zeros(shape=(P, P))
        for i in range(N):
            if y_train[i] == 0:
                sigma0 += np.transpose(X_train[i] - mu0).dot(X_train[i] - mu0)
            elif y_train[i] == 1:
                sigma1 += np.transpose(X_train[i] - mu1).dot(X_train[i] - mu1)

        # Now calculate the Sw & Sb for the LDA process
        Sw = sigma0 + sigma1
        Sb = np.transpose(mu1 - mu0).dot(mu1 - mu0)

        # Now the beta is the eigenvalue of the matrix Sw-1Sb
        self.beta = np.linalg.inv(Sw).dot(np.transpose(mu1 - mu0))
        print("Intra class variance is", ((mu1 - mu0).dot(self.beta))**2, "inter class variance is", \
                np.transpose(self.beta).dot(sigma1).dot(self.beta) + np.transpose(self.beta).dot(sigma0).dot(self.beta))
        self.threshold = (negIndex[0].shape[0] * mu0.dot(self.beta) + posIndex[0].shape[0] * mu1.dot(self.beta)) / N

    def predict(self, X):
        """
        Predict the input X's label accroding to the trained model

        """
        pred = X.dot(self.beta)
        pred[pred > self.threshold] = 1
        pred[pred < self.threshold] = 0
        return pred

def sklearnLDA(X, y):
    """
    SKLearn LDA model used to compare with mine

    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    return lda
