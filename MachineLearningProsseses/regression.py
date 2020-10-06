from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LinearRegression


class Regression:
    def __init__(self, clf_type="linear"):
        print("")
        if clf_type == "linear":
            self.clf = LinearRegression(n_jobs=-1)
            self.clf_name = "LinearRegression"
        elif clf_type == "SVR":
            self.clf = svm.SVR()
            self.clf_name = "SVR"
        else:
            print("Error: Uknown Classifier Type method. Classifier set to default: LinearRegression")
            self.clf = LinearRegression(n_jobs=-1)
            self.clf_name = "LinearRegression"

    def train(self, x, y):
        x = preprocessing.scale(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        self.clf.fit(x_train, y_train)
        acc = self.clf.score(x_test, y_test)
        print(self.clf_name + "_acc = %0.3f" % acc)

