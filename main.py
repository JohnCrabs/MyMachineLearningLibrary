import sys
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


def loadingBar(count, total, size):
    percent = float(count)/float(total)*100
    sys.stdout.flush()
    sys.stdout.write("\r" + str(int(count)).rjust(3, '0') + "/" +
                     str(int(total)).rjust(3, '0') + ' [' + '=' * int(percent/10)*size
                     + ' ' * (10 - int(percent / 10)) * size + ']')


def run_regression_example():
    # ------------------------------------ #
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

    # ------------------------------------ #

    reg_clf = reg.Regression("LinearRegression")
    # ------------------------------------------------ #
    # If these lines not commented: Train CLF and Export the trained CLF to file
    reg_clf_acc = reg_clf.train(x, y)
    reg_clf.io_clf("Data/clf/reg_clf", import_clf=False)  # Change the path to an existing to work
    # ------------------------------------------------ #
    # linreg.io_clf("Data/clf/linreg.clf", import_clf=True)  # Comment lines above and uncomment this (import clf)
    # ------------------------------------------------ #
    linreg_predic = reg_clf.predict(x_lately)
    print(linreg_predic, "%0.3f" % reg_clf_acc, forecast_out)
    df['Forecast_LinReg'] = np.nan
    df_fill_nan(df, linreg_predic)

    df['Close'].plot()
    df['Forecast_LinReg'].plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()


def run_regression_for_covid():
    run_tests = 1000000

    covid_dataset = pd.read_csv('Data/Covid/gihpwb.csv')
    '''
    df = [covid_dataset['ID'],
          covid_dataset['REGION'],
          covid_dataset['Area_km2'],
          covid_dataset['NDVI'],
          covid_dataset['NDVI_StDev'],
          covid_dataset['NDVI_StDev_km2'],
          covid_dataset['NDVI_div_km2'],
          covid_dataset['COVID-19_15Aug2020_Last_10_Days'],
          covid_dataset['COVID-19_31Aug2020_Last_10_Days'],
          covid_dataset['COVID-19_15Sep2020_Last_10_Days'],
          covid_dataset['COVID-19_30Sep2020_Last_10_Days'],
          covid_dataset['COVID-19_15Oct2020_Last_10_Days'],
          covid_dataset['COVID-19_30Oct2020_Last_10_Days'],
          covid_dataset['COVID-19_15Nov2020_Last_14_Days'],
          covid_dataset['COVID-19_30Nov2020_Last_14_Days']]
    '''

    # print(covid_dataset.keys())
    df_NDVI = [covid_dataset['NDVI_StDev_km2'].tolist(), covid_dataset['NDVI_div_km2'].tolist()]
    # df_NDVI = [covid_dataset['NDVI_StDev_km2'].tolist()]
    df_Covid_int = [covid_dataset['COVID-19_15Aug2020_Last_10_Days'].tolist(),
                    covid_dataset['COVID-19_31Aug2020_Last_10_Days'].tolist(),
                    covid_dataset['COVID-19_15Sep2020_Last_10_Days'].tolist(),
                    covid_dataset['COVID-19_30Sep2020_Last_10_Days'].tolist(),
                    covid_dataset['COVID-19_15Oct2020_Last_10_Days'].tolist(),
                    covid_dataset['COVID-19_30Oct2020_Last_10_Days'].tolist(),
                    covid_dataset['COVID-19_15Nov2020_Last_14_Days'].tolist(),
                    covid_dataset['COVID-19_30Nov2020_Last_14_Days'].tolist()]
    df_Covid_float = [(covid_dataset['COVID-19_15Aug2020_Last_10_Days'] / 4.0).tolist(),
                      (covid_dataset['COVID-19_31Aug2020_Last_10_Days'] / 4.0).tolist(),
                      (covid_dataset['COVID-19_15Sep2020_Last_10_Days'] / 4.0).tolist(),
                      (covid_dataset['COVID-19_30Sep2020_Last_10_Days'] / 4.0).tolist(),
                      (covid_dataset['COVID-19_15Oct2020_Last_10_Days'] / 4.0).tolist(),
                      (covid_dataset['COVID-19_30Oct2020_Last_10_Days'] / 4.0).tolist(),
                      (covid_dataset['COVID-19_15Nov2020_Last_14_Days'] / 4.0).tolist(),
                      (covid_dataset['COVID-19_30Nov2020_Last_14_Days'] / 4.0).tolist()]

    df_x = []
    df_y_float = []
    df_y_int = []
    for i in range(0, 6):
        list_tmp = [df_NDVI[0], df_NDVI[1], df_Covid_float[i], df_Covid_float[i + 1]]
        # list_tmp = [df_NDVI[0], df_Covid[i], df_Covid[i + 1]]
        df_x.append(list_tmp)
        df_y_float.append(df_Covid_float[i + 2])
        df_y_int.append(df_Covid_int[i + 2])

    df_x = np.array(df_x)
    df_x = np.concatenate(df_x, axis=1).T
    df_y_float = np.concatenate(df_y_float, axis=0)
    df_y_int = np.concatenate(df_y_int, axis=0)

    reg_clf = reg.Regression("LinearRegression", test_size=0.2)
    # reg_clf = reg.Regression("SVM_SVR")
    # ------------------------------------------------ #
    # If these lines not commented: Train CLF and Export the trained CLF to file
    reg_clf_acc = 0.0
    reg_check_acc = 0.6
    reg_check_counter = 0
    reg_check_acc_max = 0
    clf = None

    # print(df_x)
    # print(df_y)

    print("Run %i Regression iterations." % run_tests)
    for index in range(run_tests):
        loadingBar(count=index+1, total=run_tests, size=2)
        acc_tmp = reg_clf.train(df_x, df_y_float)
        reg_clf_acc += acc_tmp
        if acc_tmp >= reg_check_acc:
            # print(acc_tmp)
            reg_check_counter += 1
            if acc_tmp > reg_check_acc_max:
                clf = reg_clf.ret_clf()
                reg_check_acc_max = acc_tmp

    reg_clf.set_clf(clf)
    export_path = "Data/clf/reg/reg_covid_clf_acc_" + "%0.3f" % reg_check_acc_max

    reg_clf.io_clf(export_path, import_clf=False)  # Change the path to an existing to work
    print("")
    print("Regression_Acc = %0.3f" % (reg_clf_acc / run_tests))
    print("Acc >= %0.3f" % reg_check_acc + " for %i" % reg_check_counter + " out of %i iterations." % run_tests)
    print("Max check accuracy: %0.3f\n" % reg_check_acc_max)

    classif_clf = classif.Classification('SVC')
    # ------------------------------------------------ #
    # If these lines not commented: Train CLF and Export the trained CLF to file
    classif_clf_acc = 0.0
    classif_check_acc = 0.6
    classif_check_counter = 0
    classif_check_acc_max = 0
    clf = None

    print("Run %i Classification iterations." % run_tests)

    for index in range(run_tests):
        loadingBar(count=index+1, total=run_tests, size=2)
        acc_tmp = classif_clf.train(df_x, df_y_int)
        classif_clf_acc += acc_tmp
        if acc_tmp >= classif_check_acc:
            # print(acc_tmp)
            classif_check_counter += 1
            if acc_tmp > classif_check_acc_max:
                clf = reg_clf.ret_clf()
                classif_check_acc_max = acc_tmp

    classif_clf.set_clf(clf)
    export_path = "Data/clf/classif/classif_covid_clf_acc_" + "%0.3f" % classif_check_acc_max

    reg_clf.io_clf(export_path, import_clf=False)  # Change the path to an existing to work
    print("")
    print("Classification_Acc = %0.3f" % (classif_clf_acc / run_tests))
    print("Acc >= %0.3f" % classif_check_acc + " for %i" % classif_check_counter + " out of %i iterations." % run_tests)
    print("Max check accuracy: %0.3f\n" % classif_check_acc_max)


def run_classification_example_knn():
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
        for index in range(len(full_data)):
            full_data[index].append(ys[index])
        random.shuffle(full_data)

        train_set = {2: [], 4: []}
        test_set = {2: [], 4: []}
        d_train = full_data[:-int(test_size * len(full_data))]
        d_test = full_data[-int(test_size * len(full_data)):]

        for index in d_train:
            train_set[index[-1]].append(index[:-1])
        for index in d_test:
            test_set[index[-1]].append(index[:-1])
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


def run_classification_example_SVM_SVC():
    run_tests = 25

    df = pd.read_csv('./Data/BreastCancerClassificationData/breast-cancer-wisconsin.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    x = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    svc = classif.Classification('SVC')
    # ------------------------------------------------ #
    # If these lines not commented: Train CLF and Export the trained CLF to file
    svc_accuracy = []
    for i in range(run_tests):
        svc_acc = svc.train(x, y)
        svc_accuracy.append(svc_acc)
    svc.io_clf("Data/clf/knn", import_clf=False)  # Change the path to an existing to work
    # ------------------------------------------------ #
    # knn.io_clf("Data/clf/knn.clf", import_clf=True)  # Comment lines above and uncomment this (import clf)
    # ------------------------------------------------ #
    # linreg_predic = knn.predict()
    svc_acc = sum(svc_accuracy) / len(svc_accuracy)
    print("Scikit-Learn Accuracy = %0.3f" % svc_acc)

    # example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
    # example_measures = example_measures.reshape(len(example_measures), -1)
    # example_prediction = svc.predict(example_measures)
    # print(example_prediction)


# Use scikit-learn algorithms
# run_regression_example()
# run_classification_example_knn()
# run_classification_example_SVM_SVC()

# Use my algorithms
# bfs = recreg.RecRegression()
# bfs.LinearRegression()

run_regression_for_covid()
