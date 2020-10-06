import math
import pandas as pd
import quandl
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

from MachineLearningProsseses import regression as reg
from MachineLearningRecreateProsseses import rec_regression as recreg

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
    x = x[:-forecast_out]
    x_lately = x[-forecast_out:]
    df.dropna(inplace=True)
    y = np.array(df['label'])

    linreg = reg.Regression("linear")
    # ------------------------------------------------ #
    # If these lines not commented: Train CLF and Export the trained CLF to file
    linreg_acc = linreg.train(x, y)
    linreg.io_clf("data/clf/linreg", import_clf=False)  # Change the path to an existing to work
    # ------------------------------------------------ #
    # linreg.io_clf("data/clf/linreg.clf", import_clf=True)  # Comment lines above and uncomment this (import clf)
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


# run_regression_example()

bfs = recreg.RecRegression()
bfs.LinearRegression()
