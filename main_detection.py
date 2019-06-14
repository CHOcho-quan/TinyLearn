import numpy as np
import cv2, os, pickle, keras, glob
import matplotlib.pyplot as plt

from utils import *
from models.FisherModel import MyFisherModel, sklearnLDA
from models.LogisticRegressioner import MyLogisticRegression, SKLearnLogisticRegression

def NMS(rects, l):
    """
    Do NMS compression of the given rects of the faces

    """
    new_rects = []
    flag = True
    total_s = l**2
    print(len(rects))
    for i in range(len(rects)):
        flag = True
        if i == 0:
            new_rects.append(rects[i])
            continue

        for rect in new_rects:
            fx = max(rect[0], rects[i][0])
            fy = max(rect[2], rects[i][2])
            sx = min(rect[1], rects[i][1])
            sy = min(rect[3], rects[i][3])

            if sy >= fy and sx >= fx and (sy - fy) * (sx - fx) > total_s / 2:
                flag = False
                break
        if flag:
            new_rects.append(rects[i])

    return new_rects

def detectSingleScale(img, l, stride, model):
    """
    Detecting single scale face of the scaled image
    img : input scaled image
    l : the height & width of the sliding window
    model : model used to predict the face

    """
    rects = []
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(0, img.shape[1] - l, stride):
        for j in range(0, img.shape[0] - l, stride):
            print(i, j)
            cut = cv2.resize(image[j:j+l, i:i+l], (96, 96))
            # cv2.imshow("cut", cut)
            # cv2.waitKey(0)
            hog = cv2.HOGDescriptor("hog.xml")
            feats = hog.compute(img=cut, winStride=None, padding=None, locations=None).reshape(1, 900)
            # print(feats)
            pred = model.predict(feats)
            print(pred)
            if pred >= 0.5:
                rects.append([i, i+l, j, j+l])

    # rects = NMS(rects, l)
    for rect in rects:
        x1, x2, y1, y2 = rect
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imshow("r", img)
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
    # X_train = dataNormalize(X_train)
    # X_test = dataNormalize(X_test)
    X_train, y_train = shuffle(X_train, y_train)

    # Fisher model part
    # myFisherModel = MyFisherModel()
    # myFisherModel.fit(X_train, y_train)
    # myPred = myFisherModel.predict(X_test)
    # myAccuracy = np.count_nonzero(myPred.reshape(myPred.shape[0], 1) == y_test) / y_test.shape[0]
    # print("LDA", myAccuracy)

    # Logistic Regression part
    sklearnRegressioner = SKLearnLogisticRegression(1, 100, X_train, y_train)
    skPred = sklearnRegressioner.predict(X_test)
    skAccuracy = np.count_nonzero(skPred.reshape(skPred.shape[0], 1) == y_test) / y_test.shape[0]
    print("Logisitic", skAccuracy)

    for pics in originalPics:
        img = cv2.imread(pics)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        detectMultiScale(img, 96, 6, [0.5, 0.6, 0.7], sklearnRegressioner)
