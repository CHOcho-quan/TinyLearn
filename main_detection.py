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
            # feats -= np.mean(feats)
            # feats /= np.std(feats)
            # print(feats)
            pred = np.squeeze(model.predict(feats))
            print(pred)
            if pred == 1:
                rects.append([i, i+l, j, j+l])

    rects = NMS(rects, l)
    for rect in rects:
        x1, x2, y1, y2 = rect
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
    cv2.imshow("r", img)
    cv2.waitKey(0)

def detectSingleScaleModels(img, l, stride, models):
    """
    Detect single scale with multi models

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
            # feats -= np.mean(feats)
            # feats /= np.std(feats)
            pred = 1
            for model in models:
                pred = (np.squeeze(model.predict(feats)) and pred)
                print("pred", pred)
            if pred == 1:
                rects.append([i, i+l, j, j+l])

    rects = NMS(rects, l)
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

def detectMultiModel(img, l, stride, scales, models):
    """
    Detecting image by multi models to get more accurate result

    """
    for scale in scales:
        tmp = cv2.resize(src=img, dsize=(0, 0), fx=scale, fy=scale)
        detectSingleScaleModels(tmp, l, stride, models)

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
    myFisherModel = MyFisherModel()
    myFisherModel.fit(X_train, y_train)
    myPred = myFisherModel.predict(X_test)
    myAccuracy = np.count_nonzero(myPred.reshape(myPred.shape[0], 1) == y_test) / y_test.shape[0]
    print("LDA", myAccuracy)

    # Logistic Regression part
    # mySGDRegressioner = MyLogisticRegression(optimization = 'sgd')
    # print("My Regressioner doing Logisitic Regression")
    # mySGDRegressioner.fit(X_train, y_train)
    sklearnRegressioner = SKLearnLogisticRegression(1, 100, X_train, y_train)
    skPred = sklearnRegressioner.predict(X_test)
    skAccuracy = np.count_nonzero(skPred.reshape(skPred.shape[0], 1) == y_test) / y_test.shape[0]
    print("Logisitic", skAccuracy)

    # SVM part
    # sklearnSVMRegressioner = sklearnSVM(X_train, y_train, kernel='rbf')

    # CNN part
    # X_train, y_train, X_val, y_val = splitData(X_train, y_train, 0.3)
    #
    # model = Sequential()
    # model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
    # model.add(BatchNormalization())
    # model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D())
    # model.add(Flatten())
    # model.add(Dense(128, activation='sigmoid'))
    # model.add(Dense(2, activation='softmax'))
    # model.compile(optimizer='adam', loss='mse')
    # model.summary()
    #
    # callback = EarlyStopping(monitor="loss", patience=0.01, verbose=1, mode="auto")
    # model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_val, y_val), callbacks=[callback])

    for pics in originalPics:
        img = cv2.imread(pics)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        detectMultiScale(img, 96, 6, [0.4, 0.5, 0.6], sklearnRegressioner)
        # detectMultiModel(img, 96, 6, [0.4, 0.5, 0.6], [sklearnRegressioner, myFisherModel])
