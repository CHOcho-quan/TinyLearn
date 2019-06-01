import numpy as np
import cv2, os, pickle, keras, glob
import matplotlib.pyplot as plt

from utils import *
from models.FisherModel import MyFisherModel, sklearnLDA
from models.LogisticRegressioner import MyLogisticRegression

def detectSingleScale(img, l, stride, model):
    """
    Detecting single scale face of the scaled image
    img : input scaled image
    l : the height & width of the sliding window
    model : model used to predict the face

    """
    rects = []
    for i in range(0, img.shape[1] - l, stride):
        for j in range(0, img.shape[0] - l, stride):
            print(i, j)
            cut = img[j:j+l, i:i+l]
            # cv2.imshow("cut", cut)
            # cv2.waitKey(0)
            hog = cv2.HOGDescriptor("hog.xml")
            feats = np.transpose(np.array(hog.compute(img=cut, winStride=None, padding=None, locations=None)))
            pred = model.predict(feats)
            print(pred)
            if pred >= 0.5:
                rects.append([i, i+l, j, j+l])

    for rect in rects:
        x1, x2, y1, y2 = rect
        result = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
        cv2.imshow("r", result)
        cv2.waitKey(0)

def detectMultiScale(img, l, stride, scales, model):
    """
    Detecting image scaled by the scales array
    img : input image
    l : the height & width of the sliding window
    sclaes : given scales to be resized
    model : model used to predict the face

    """
    for scale in scales:
        tmp = cv2.resize(src=img, dsize=(0, 0), fx=scale, fy=scale)
        detectSingleScale(tmp, l, stride, model)

if __name__=='__main__':
    originalPics = glob.glob('./originalPics/*/*/*/*/*.jpg')

    # Getting the data
    X_train, y_train = getData()
    X_test, y_test = getData(type='test')
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    X_train = dataNormalize(X_train)
    X_test = dataNormalize(X_test)
    X_train, y_train = shuffle(X_train, y_train)

    # Logistic Regression part
    mySGDRegressioner = MyLogisticRegression(optimization = 'sgd')
    print("My Regressioner doing Logisitic Regression")
    mySGDRegressioner.fit(X_train, y_train)

    for pics in originalPics:
        img = cv2.imread(pics)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        detectMultiScale(img, 96, 6, [0.7, 1, 1.3], mySGDRegressioner)
