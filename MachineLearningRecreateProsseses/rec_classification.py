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
