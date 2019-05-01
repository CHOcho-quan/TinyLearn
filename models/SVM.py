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
        self.loss_history = None
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
            return np.exp(np.sum((X - Y)**2) / 2)
        return np.exp(-np.sum((X-Y)**2, axis=1) / 2)

    def _cosKernel(self, X, Y):
        """
        Cosine similarity Kernel calculating with matrixes or vectors X and Y

        """
        return np.transpose(X).dot(Y) / np.sqrt(np.sum(X**2))*np.sqrt(np.sum(Y**2))

    def fit(self, X_train, y_training):
        """
        Train the linear SVM with training data X_train & y_train
        X_train : Input training features with shape n x p
        y_train : Input label with the shape n x 1

        """
        N, P = X_train.shape
        self.bias = 0
        self.alpha = np.zeros(shape=(N, 1))
        self.loss_history = 0
        loss = 0
        y_train = y_training.copy()
        y_train[y_train==0] = -1

        # Calculate the wX + b
        wX = np.sum(self.alpha * y_train * self.innerProduct(np.transpose(X_train), np.transpose(X_train)), axis=1).reshape(N, -1)
        pred = wX + self.bias

        b = - 1 / 2 * (np.min(wX[y_train==1]) + np.max(wX[y_train==-1]))
        # Implementing SMO algorithm to refresh b & alpha here
        iter = 0
        L = np.zeros_like(self.alpha)
        H = np.zeros_like(self.alpha)
        H[:, :] = self.C
        while (iter < self.max_iter): # OR Converge
            for i in range(self.alpha.shape[0]):
                alphai = self.alpha[i, 0]
                if y_train[i] * (wX[i] + b) > 1 and alphai == L[i]:
                    continue
                if y_train[i] * (wX[i] + b) < 1 and alphai == H[i]:
                    continue
                if y_train[i] * (wX[i] + b) == 1 and alphai < H[i] and alphai > L[i]:
                    continue

                # Now the alpha need to be updated by the SMO process, select alphaj by maximizing |Ei - Ej|
                E = pred - y_train
                Ei = E[i]
                deltaE = np.abs(E - Ei)
                Ej = np.max(deltaE)
                idx = np.where(deltaE == Ej)[0][0]
                alphaj = self.alpha[idx, 0]

                # The two alphas are in the same side
                if (y_train[i] == y_train[idx]):
                    Lj = max(0, alphai + alphaj - self.C)
                    Hj = min(self.C, alphai + alphaj)
                else:
                    Lj = max(0, alphaj - alphai)
                    Hj = min(self.C, self.C - alphai + alphaj)

                if (Lj == Hj):
                    continue
                L[idx, 0] = Lj
                H[idx, 0] = Hj
                eta = self.innerProduct(X_train[i, :], X_train[i, :]) + self.innerProduct(X_train[idx, :], X_train[idx, :]) - 2 * self.innerProduct(X_train[i, :], X_train[idx, :])
                if eta < 0:
                    continue

                alphaj_ori = alphaj.copy()
                alphai_ori = alphai.copy()
                alphaj = alphaj + y_train[idx] * (Ei - Ej) / eta
                alphai = alphai + y_train[idx] * y_train[i] * (alphaj_ori - alphaj)
                self.alpha[i, 0] = alphai
                self.alpha[idx, 0] = alphaj

                b1 = -Ei + y_train[i] * self.innerProduct(X_train[i, :], X_train[i, :]) * (alphai - alphai_ori) \
                    - y_train[idx] * self.innerProduct(X_train[i, :], X_train[idx, :]) * (alphaj - alphaj_ori) + b
                b2 = -Ej + y_train[i] * self.innerProduct(X_train[i, :], X_train[idx, :]) * (alphai - alphai_ori) \
                    - y_train[idx] * self.innerProduct(X_train[idx, :], X_train[idx, :]) * (alphaj - alphaj_ori) + b
                if (alphai > 0 and alphai < self.C):
                    b = b1
                elif (alphaj > 0 and alphaj < self.C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
            iter+=1
            print("Iteration {0}: ".format(iter), np.count_nonzero(self.alpha))
        self.w = np.sum(self.alpha * y_train * X_train, axis=0).reshape(P, -1)
        self.bias = b

    def predict(self, X):
        """
        Make Predictions based on the learned w & bias

        """
        pred = X.dot(self.w) + self.bias
        pred[pred < 1] = 0
        pred[pred >= 1] = 1
        return pred

def sklearnSVM(X_train, y_train, C = 1.0, max_iter = 100):
    """
    SKLearn SVM to be tested with MySVM results
    C : The penalty term of SVM
    max_iter : Max hard iteration of the SVM

    """
    clf = SVC(C=C, max_iter=max_iter).fit(X_train, y_train)
    return clf
