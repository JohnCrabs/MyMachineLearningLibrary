from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression


class Regression:
    def __init__(self, clf_type="linear"):
        print("")
        if clf_type == "linear":
            self.clf = LinearRegression(n_jobs=-1)
        elif clf_type == "SVR":
            self.clf = svm.SVR()
        else:
            print("Error: Uknown Classifier Type method. Classifier set to default: LinearRegression")
            self.clf = LinearRegression(n_jobs=-1)

    def train(self, x, y):
        # x = preprocessing.scale(x)
        # y = preprocessing.scale(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        self.clf.fit(x_train, y_train)
        acc = self.clf.score(x_test, y_test)
        return acc

    def predict(self, x):
        return self.clf.predict(x)
