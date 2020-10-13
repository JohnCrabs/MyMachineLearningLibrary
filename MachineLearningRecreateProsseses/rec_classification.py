from math import sqrt
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import numpy as np

style.use('fivethirtyeight')


class SupportVectorMachine:
    def __init__(self, visualization=True):
        self.data = None
        self.max_feature_value = None
        self.min_feature_value = None
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001]
        # extremely expensive
        b_range_multiple = 5
        #
        b_multiple = 5
        #
        latest_optimum = self.max_feature_value*10

        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                pass
            
    # predict
    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)


class RecClassification:
    def __init__(self):
        pass

    @staticmethod
    def eucledian_distance(p1, p2):
        d_sum = 0
        for i in range(len(p1)):
            d_sum += (p1[i] - p2[i])**2
        return sqrt(d_sum)

    @staticmethod
    def manhantan_distance(p1, p2):
        d_sum = 0
        for i in range(len(p1)):
            if p1[i] - p2[i] > 0:
                d_sum += p1[i] - p2[i]
            else:
                d_sum += p2[i] - p1[i]
        return d_sum

    def knn(self, train_data):
        def k_nearest_neighbors(input_data, predict, k=3):
            if len(input_data) >= k:
                warnings.warn('K is set to a value less than total voting groups!')
            distances = []
            for group_name in input_data:
                for features in input_data[group_name]:
                    manhantan_distance = self.manhantan_distance(features, predict)
                    distances.append([manhantan_distance, group_name])

            votes = [i[1] for i in sorted(distances)[:k]]
            vote_results = Counter(votes).most_common(1)[0][0]
            confid = Counter(votes).most_common(1)[0][1] / k

            return vote_results, confid

        correct = 0
        total = 0
        sum_confidence = 0

        for group in train_data:
            for data in train_data[group]:
                vote, confidence = k_nearest_neighbors(train_data, data, k=5)
                if group == vote:
                    sum_confidence += confidence
                    correct += 1
                total += 1

        accuracy = correct / total
        confidence = sum_confidence / correct
        return accuracy, confidence

    def svm_SVC(self):
        data_dict = {-1: np.array([[1, 7],
                                   [2, 8],
                                   [3, 8]]),
                     1:np.array([[5, 1],
                                 [6, -1],
                                 7, 3])}
