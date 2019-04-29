import numpy as np
import cv2
import pickle
import random
import math
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

class MyLogisticRegression:
    """
    My Logisitic Regression Implementation

    """
    def __init__(self, regularization = 1.0, optimization = 'sgd', max_iter = 100, tolerance = 1e-5, learning_rate = 1e-3):
        """
        Initiation function
        regularization : the constant weight of regularization term
        optimization : the method of optimization, sgd or langevin
        max_iter : the hard stop criterion if not converge
        tolerance : the stop criterion of the regression

        """
        self.Lambda = regularization
        self.optimizer = optimization
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.beta = None
        self.loss_history = None
        self.lr = learning_rate

    def sigmoid(self, X):
        """
        Sigmoid function to be used in the regression
        y(x) = 1 / (1 + e^(-x))

        """
        return np.exp(X) / (1 + np.exp(X))

    def fit(self, X_train, y_train):
        """
        Fitting the model with the training data
        X_train : input training features of 900 x 1 vectors
        y_train : the label of the training features with 1 representing faces

        """
        self.beta = np.random.normal(size=(X_train.shape[1], 1))
        self.loss_history = []
        loss = None
        grad = np.zeros_like(self.beta)

        for i in tqdm(range(self.max_iter)):
            # Currently prediction
            pred = X_train.dot(self.beta)
            prediction = pred.copy()
            prediction[prediction > 0.5] = 1
            prediction[prediction < 0.5] = 0
            accuracy = np.count_nonzero(prediction == y_train) / prediction.shape[0]
            print("Round {0}".format(i), "Accuracy : ", accuracy)

            # Calculating Loss
            loss = np.sum(- y_train * pred)
            for Xi in X_train:
                loss += math.log(1 + math.exp(Xi.dot(self.beta)))
            loss += np.sum(self.Lambda * self.beta * self.beta)
            self.loss_history.append(loss)

            # Calculating the grad
            grad += np.transpose(X_train).dot(self.sigmoid(X_train.dot(self.beta)) - y_train)
            grad += 2 * self.Lambda * self.beta

            # Updating beta accroding to the grad
            if (self.optimizer == 'sgd'):
                if (random.random() >= 0.3):
                    self.beta -= self.lr * grad

            # Re-initialization
            loss = 0
            grad = np.zeros_like(self.beta)

            if (1 - accuracy < self.tolerance):
                break

    def predict(self, X_test):
        """
        Given a p x 1 sample X_test return a prediction of the given feature
        Output : Probability sigmoid(XT * beta)

        """
        pred = self.sigmoid(X_test.dot(self.beta))
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

def SKLearnLogisticRegression(Lambda, max_iter, X, y):
    """
    SKLearn implemented Logisitic Regression to compare with MyLogisticRegression
    Lambda : the regularization term
    max_iter : max hard iteration when stopping the process

    """
    clf = LogisticRegression(C = Lambda, max_iter=max_iter).fit(X, y)
    return clf
