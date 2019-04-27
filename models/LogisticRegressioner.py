import numpy as np
import cv2
import pickle
from sklearn.linear_model import LogisticRegression
from utils import *

class MyLogisticRegression:
    def __init__(self):
        pass

def SKLearnLogisticRegression(Lambda, max_iter, X, y):
    """
    SKLearn implemented Logisitic Regression to compare with MyLogisticRegression
    Lambda : the regularization term
    max_iter : max hard iteration when stopping the process

    """
    clf = LogisticRegression(C = Lambda, max_iter=max_iter).fit(X, y)
    return clf
