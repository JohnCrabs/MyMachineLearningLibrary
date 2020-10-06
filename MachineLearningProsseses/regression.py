from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression


def mll_LinearRegression(x, y):
    x = preprocessing.scale(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, )
    clf = LinearRegression(n_jobs=-1)
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print("LinearRegression_acc = %0.3f" % acc)


def mll_SVM_SVR(x, y):
    x = preprocessing.scale(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, )
    clf = svm.SVR()
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print("SVM_SVR_acc = %0.3f" % acc)
