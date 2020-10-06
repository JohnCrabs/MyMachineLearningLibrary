import pickle
import os


def int_to_string(num: int):
    return str(num).zfill(3)


def save_clf(path_filename: str, clf, suffix=".clf"):
    if os.path.isfile(path=path_filename + suffix):
        index = 0
        while os.path.isfile(path=path_filename + "_" + int_to_string(index) + suffix):
            index += 1
        path_filename += "_" + int_to_string(index)
    path_filename += suffix
    with open(path_filename, 'wb') as f:
        pickle.dump(clf, f)


def load_clf(path: str):
    if os.path.isfile(path=path):
        pickle_in = open(path, 'rb')
        return pickle.load(pickle_in)
    else:
        print("Error: File not found!")
        exit(404)
    return None