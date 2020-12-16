from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression

import warnings
from Utilities import io_clf

dictReg = {"LinearRegression": LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None),
           "SVM_SVR": svm.SVR(kernel='linear', degree=3, gamma='scale', coef0=0.0, tol=1e-1, C=1.0, shrinking=True,
                              cache_size=200, verbose=False, max_iter=10)}


class Regression:
    def __init__(self, clf_type="LinearRegression", test_size=0.2):
        """
        :param clf_type: It takes the values - "LinearRegression", SVM_SVR"
        """
        self.test_size = test_size
        if dictReg.get(clf_type) is not None:
            self.clf = dictReg[clf_type]
        else:
            warnings.warn(clf_type + " isn't an option! Default classifier will be used instead (LinearRegression)!")
            self.clf = dictReg["LinearRegression"]

    def io_clf(self, path, import_clf=False):
        if import_clf:
            self.clf = io_clf.load_clf(path=path)
        else:
            io_clf.save_clf(path_filename=path, clf=self.clf)

    def train(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size)
        self.clf.fit(x_train, y_train)
        acc = self.clf.score(x_test, y_test)
        return acc

    def predict(self, x):
        return self.clf.predict(x)
