import math
import pandas as pd
import quandl
import numpy as np

from MachineLearningProsseses import regression as reg

pd.set_option('display.max_columns', None)


def run_regression_example():
    df = quandl.get("FSE/ZO1_X", authtoken="DkzKmoFJZXxAykgE_dWy")
    df = df[['Open', 'High', 'Low', 'Close', 'Traded Volume']]
    df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100
    df['PCT_Change'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['Volume'] = df['Traded Volume']
    df = df[['Close', 'HL_PCT', 'PCT_Change', 'Volume']]

    forecast_col = 'Close'
    df.fillna(-99999, inplace=True)

    forecast_out = int(math.ceil(0.003 * len(df)))
    print(forecast_out)

    df['label'] = df[forecast_col].shift(-forecast_out)
    x = np.array(df.drop(['label'], 1))
    x = x[:-forecast_out]
    x_lately = x[-forecast_out:]
    df.dropna(inplace=True)
    y = np.array(df['label'])

    linreg = reg.Regression("linear")
    linreg.train(x, y)

    svr = reg.Regression("SVR")
    svr.train(x,y)


run_regression_example()
