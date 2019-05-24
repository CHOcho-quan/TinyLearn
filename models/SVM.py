import numpy as np
import cv2
import os
import math
import pickle
import cvxopt
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
        self.bias = 0.0
        self.w = None
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        if (kernel == 'rbf'):
            self.innerProduct = self._rbfKernel
        elif (kernel == 'linear'):
            self.innerProduct = self._inner
        elif (kernel == 'poly'):
            self.innerProduct = self._polyKernel
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
            return np.exp(-np.sum((X - Y)**2) * 0.1)
        return np.exp(-np.sum((X - Y)**2, axis=1) * 0.1)

    def _polyKernel(self, X, Y):
        """
        Polynomial Kernel calculating with matrixes or vectors X and Y

        """
        return (np.transpose(X).dot(Y) + 1)**2

    def _gramMatrix(self, X):
        """
        Calculate the gram matrix of the given training set X

        """
        print("computing gram Matrix")
        n_samples, n_features = X.shape
        if (self.innerProduct == self._inner):
            K = X.dot(np.transpose(X))
            return K
        K = np.zeros(shape=(n_samples, n_samples))
        for i, xi in enumerate(X):
            for j, xj in enumerate(X):
                K[i, j] = self.innerProduct(xi, xj)

        return K

    def fit(self, X_training, y_training):
        """
        Train the linear SVM with training data X_train & y_train
        X_train : Input training features with shape n x p
        y_train : Input label with the shape n x 1

        """
        y_train = y_training.copy()
        y_train[y_train==0] = -1
        n_samples, n_features = X_training.shape


        # Applying the Optimization problem
        # Viewing SVM as a QP optimization problem
        if os.path.exists(path="./data/QP.pickle"):
            with open('./data/QP.pickle', 'rb') as load_data:
                K, P, Q, G, h, A, b = pickle.load(load_data)
        else:
            K = self._gramMatrix(X_training)
            P = cvxopt.matrix(np.outer(y_train, y_train) * K)
            Q = cvxopt.matrix(-1 * np.ones(n_samples))

            G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h_std = cvxopt.matrix(np.zeros(n_samples))
            G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
            h_slack = cvxopt.matrix(np.ones(n_samples) * self.C)
            G = cvxopt.matrix(np.vstack((G_std, G_slack)))
            h = cvxopt.matrix(np.vstack((h_std, h_slack)))

            A = cvxopt.matrix(y_train.astype(np.float), (1, n_samples))

            # print(A.size)
            b = cvxopt.matrix(0.0)
            # with open('./data/QP.pickle', 'wb') as save_data:
            #     data_list = [K, P, Q, G, h, A, b]
            #     pickle.dump(data_list, save_data)

        # Getting the support vectors
        print("solving QP")
        solution = cvxopt.solvers.qp(P, Q, G, h, A, b)
        lagrange = np.ravel(solution['x'])
        support_vector_indices = lagrange > 1e-5
        self.alpha = lagrange[support_vector_indices]
        self.support_vectors = X_training[support_vector_indices]
        self.support_vector_labels = y_train[support_vector_indices]

        self.bias = np.mean(self.support_vector_labels - self.predict(self.support_vectors))

    def predict(self, X):
        """
        Make Predictions based on the learned w & bias

        """
        pred = []
        prediction = 0
        for i in range(X.shape[0]):
            for j in range(self.support_vectors.shape[0]):
                prediction += (self.alpha[j] * self.support_vector_labels[j] * self.innerProduct(np.transpose(X[i]), np.transpose(self.support_vectors[j]))) \
                                + self.bias
            pred.append(prediction)
            prediction = 0
        pred = np.array(pred)
        pred[pred > 0] = 1
        pred[pred <= 0] = 0
        return pred

def sklearnSVM(X_train, y_train, C = 1.0, max_iter = 100):
    """
    SKLearn SVM to be tested with MySVM results
    C : The penalty term of SVM
    max_iter : Max hard iteration of the SVM

    """
    clf = SVC(C=C, max_iter=max_iter).fit(X_train, y_train)
    return clf
