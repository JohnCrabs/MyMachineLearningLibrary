import math
import pandas as pd
import quandl
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import random

from MachineLearningProsseses import regression as reg
from MachineLearningProsseses import classification as classif

# from MachineLearningRecreateProsseses import rec_regression as recreg
from MachineLearningRecreateProsseses import rec_classification as recclassif

pd.set_option('display.max_columns', None)
style.use("ggplot")


def run_regression_example():
    def df_fill_nan(df_p, predict):
        last_date = df_p.iloc[-1].name
        last_unix_value = last_date.timestamp()
        one_day = 86400
        next_unix = last_unix_value + one_day
        for i in predict:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_day
            df_p.loc[next_date] = [np.nan for _ in range(len(df_p.columns) - 1)] + [i]

    df = quandl.get("FSE/ZO1_X", authtoken="DkzKmoFJZXxAykgE_dWy")
    df = df[['Open', 'High', 'Low', 'Close', 'Traded Volume']]
    df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100
    df['PCT_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['Volume'] = df['Traded Volume']
    df = df[['Close', 'HL_PCT', 'PCT_Change']]

    forecast_col = 'Close'
    df.fillna(-99999, inplace=True)

    forecast_out = int(math.ceil(0.01 * len(df)))

    df['label'] = df[forecast_col].shift(-forecast_out)
    x = np.array(df.drop(['label'], 1))
    x_lately = x[-forecast_out:]
    x = x[:-forecast_out]
    df.dropna(inplace=True)
    y = np.array(df['label'])

    linreg = reg.Regression("linear")
    # ------------------------------------------------ #
    # If these lines not commented: Train CLF and Export the trained CLF to file
    linreg_acc = linreg.train(x, y)
    linreg.io_clf("Data/clf/linreg", import_clf=False)  # Change the path to an existing to work
    # ------------------------------------------------ #
    # linreg.io_clf("Data/clf/linreg.clf", import_clf=True)  # Comment lines above and uncomment this (import clf)
    # ------------------------------------------------ #
    linreg_predic = linreg.predict(x_lately)
    print(linreg_predic, "%0.3f" % linreg_acc, forecast_out)
    df['Forecast_LinReg'] = np.nan
    df_fill_nan(df, linreg_predic)

    '''
    svr = reg.Regression("SVR")
    svr_acc = svr.train(x, y)
    svr_predic = svr.predict(x_lately)
    print(svr_predic, "%0.3f" % svr_acc, forecast_out)
    df['Forecast_SVR'] = np.nan
    df_fill_nan(df, svr_predic)
    '''

    df['Close'].plot()
    df['Forecast_LinReg'].plot()
    # df['Forecast_SVR'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def run_classification_example():
    run_tests = 25

    df = pd.read_csv('./Data/BreastCancerClassificationData/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    x = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    knn = classif.Classification('knn')
    # ------------------------------------------------ #
    # If these lines not commented: Train CLF and Export the trained CLF to file
    knn_accuracy = []
    for i in range(run_tests):
        knn_acc = knn.train(x, y)
        knn_accuracy.append(knn_acc)
    knn.io_clf("Data/clf/knn", import_clf=False)  # Change the path to an existing to work
    # ------------------------------------------------ #
    # knn.io_clf("Data/clf/knn.clf", import_clf=True)  # Comment lines above and uncomment this (import clf)
    # ------------------------------------------------ #
    # linreg_predic = knn.predict()
    knn_acc = sum(knn_accuracy) / len(knn_accuracy)
    print("Scikit-Learn Accuracy = %0.3f" % knn_acc)

    # example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
    # example_measures = example_measures.reshape(len(example_measures), -1)
    # example_prediction = knn.predict(example_measures)
    # print(example_prediction)

    # Test the knn using my written algorithm
    def train_test_split(xs, ys, test_size=0.2):
        """
        This function needs to be changed according to the dataset, because it heavilly depends on the classes.
        The custom classification algorithm uses dictionary for classification and so the data needs to be edited
        appropriately.
        :param xs: The input array.
        :param ys: The output array (indexes of classes).
        :param test_size: The percentage of the dataset size that will be use for testing purposes.
        :return:
        """
        xs = xs.astype(float).tolist()
        ys = ys.astype(float).tolist()
        full_data = xs
        for i in range(len(full_data)):
            full_data[i].append(ys[i])
        random.shuffle(full_data)

        train_set = {2: [], 4: []}
        test_set = {2: [], 4: []}
        d_train = full_data[:-int(test_size * len(full_data))]
        d_test = full_data[-int(test_size * len(full_data)):]

        for i in d_train:
            train_set[i[-1]].append(i[:-1])
        for i in d_test:
            test_set[i[-1]].append(i[:-1])
        return train_set, test_set

    bfs = recclassif.RecClassification()
    custom_knn_accuracies = []
    custom_knn_confidences = []

    for i in range(run_tests):
        train_data, test_data = train_test_split(x, y, test_size=0.4)
        custom_knn_acc, custom_knn_conf = bfs.knn(train_data)
        custom_knn_accuracies.append(custom_knn_acc)
        custom_knn_confidences.append(custom_knn_conf)

    custom_knn_acc = sum(custom_knn_accuracies) / len(custom_knn_accuracies)
    custom_knn_conf = sum(custom_knn_confidences) / len(custom_knn_confidences)
    print("Custom KNN Accuracy = %0.3f" % custom_knn_acc)
    print("Custom KNN Confidence = %0.3f" % custom_knn_conf)


# Use scikit-learn algorithms
# run_regression_example()
run_classification_example()

# Use my algorithms
# bfs = recreg.RecRegression()
# bfs.LinearRegression()
