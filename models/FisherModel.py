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

        mu0 = np.mean(X_train[negIndex[1], :], axis=0).reshape(1, P)
        mu1 = np.mean(X_train[posIndex[1], :], axis=0).reshape(1, P)

        sigma0 = np.cov(np.transpose(X_train[negIndex[1], :]))
        sigma1 = np.cov(np.transpose(X_train[posIndex[1], :]))

        # Now calculate the Sw & Sb for the LDA process
        Sw = negIndex[1].shape[0] * sigma0 + posIndex[1].shape[0] * sigma1
        Sw += np.eye(P) * 1e-18

        # Now the beta is the eigenvalue of the matrix Sw-1Sb
        self.beta = np.linalg.inv(Sw).dot(np.transpose(mu1 - mu0))
        print("Intra class variance is", ((mu1 - mu0).dot(self.beta))**2, "inter class variance is", \
                np.transpose(self.beta).dot(sigma1).dot(self.beta) + np.transpose(self.beta).dot(sigma0).dot(self.beta))
        self.threshold = (mu0.dot(self.beta) + mu1.dot(self.beta)) / 2
        # print(mu0.dot(self.beta) > self.threshold)

    def predict(self, X):
        """
        Predict the input X's label accroding to the trained model

        """
        pred = X.dot(self.beta)
        pred[pred >= self.threshold] = 1
        pred[pred < self.threshold] = 0
        return pred

def sklearnLDA(X, y):
    """
    SKLearn LDA model used to compare with mine

    """
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    return lda
