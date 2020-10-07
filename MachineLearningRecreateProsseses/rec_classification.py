import numpy as np
from math import sqrt
import warnings
from matplotlib import style
from collections import Counter


style.use('fivethirtyeight')


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
    def knn(train_data, test_data):
        def k_nearest_neighbors(data, predict, k=3):
            if len(data) >= k:
                warnings.warn('K is set to a value less than total voting groups!')
            distances = []
            for group in data:
                for features in data[group]:
                    euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
                    distances.append([euclidean_distance, group])

            votes = [i[1] for i in sorted(distances)[:k]]
            vote_results = Counter(votes).most_common(1)[0][0]

            return vote_results

        correct = 0
        total = 0

        for group in train_data:
            for data in train_data[group]:
                vote = k_nearest_neighbors(train_data, data, k=5)
                if group == vote:
                    correct += 1
                total += 1

        return correct / total
