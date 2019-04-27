import numpy as np
import cv2
from sklearn.svm import SVC
import pickle
from utils import *

class mySVM:

    def __init__(self):
        pass

def sklearnSVM(X_train, y_train, C, max_iter):
    """
    SKLearn SVM to be tested with MySVM results
    C : The penalty term of SVM
    max_iter : Max hard iteration of the SVM

    """
    clf = SVC(C=C, max_iter=max_iter).fit(X_train, y_train)
    return clf
