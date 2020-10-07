from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

from Utilities import io_clf


class Classification:
    def __init__(self, clf_type='knn'):
        if clf_type == 'knn':
            self.clf = neighbors.KNeighborsClassifier()
        else:
            print("Error: Uknown Classifier Type method. Classifier set to default: KNeighborsClassifier")
            self.clf = neighbors.KNeighborsClassifier()

    def io_clf(self, path, import_clf=False):
        if import_clf:
            self.clf = io_clf.load_clf(path=path)
        else:
            io_clf.save_clf(path_filename=path, clf=self.clf)

    def train(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
        self.clf.fit(x_train, y_train)
        acc = self.clf.score(x_test, y_test)
        return acc

    def predict(self, x):
        return self.clf.predict(x)