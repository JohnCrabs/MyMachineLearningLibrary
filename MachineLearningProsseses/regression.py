import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression


def mll_LinearRegression(input, output):
    x = preprocessing.scale(input)
    y = output

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, )
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print("%0.3f" % acc)


def mll_SVM_SVR(input, output):
    x = preprocessing.scale(input)
    y = output

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, )
    clf = svm.SVR()
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print("%0.3f" % acc)
