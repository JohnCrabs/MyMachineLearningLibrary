from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random

from matplotlib import style

style.use('fivethirtyeight')


class RecRegression:
    def __init__(self):
        self.xs, self.ys = self.create_datatest(hm=40, variance=20, step=2, correlation='pos')

    @staticmethod
    def create_datatest(hm, variance, step=2, correlation='pos'):
        val = 1
        ys = []
        for i in range(hm):
            y = val + random.randrange(-variance, variance)
            ys.append(y)

            if correlation and correlation == "pos":
                val += step
            elif correlation and correlation == "pos":
                val -= step
        xs = [i for i in range(len(ys))]
        return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

    def LinearRegression(self):
        def best_fit_slope_and_intercept(xs, ys):
            m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
                 ((mean(xs) ** 2) - mean(xs ** 2)))
            b = mean(ys) - m*mean(xs)

            return m, b

        def squared_error(ys_real, ys_predict):
            return sum((ys_predict - ys_real)**2)

        def coefficient_of_determination(ys_real, ys_predict):
            y_mean_line = np.array([mean(ys_real) for _ in ys_real])
            sq_error_regr = squared_error(ys_real, ys_predict)
            sq_error_y_mean = squared_error(ys_real, y_mean_line)
            return 1 - (sq_error_regr / sq_error_y_mean)

        m, b = best_fit_slope_and_intercept(self.xs, self.ys)
        regression_line = np.array([(m*x)+b for x in self.xs])

        r2 = coefficient_of_determination(self.ys, regression_line)
        print(r2)

        plt.scatter(self.xs, self.ys)
        plt.plot(self.xs, regression_line)
        plt.show()
