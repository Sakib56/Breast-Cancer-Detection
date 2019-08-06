import numpy as np
from collections import Counter

class KNearestNeighbor():
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def predict(self, data, sample, k=3):
        distances = [[np.linalg.norm(np.array(features)-np.array(sample)), diagnosis]
                     for diagnosis in data for features in data[diagnosis]]

        votes = [di[1] for di in sorted(distances)[:k]]
        verdict = Counter(votes).most_common(1)[0][0]

        return verdict


# dataset = {"A": [[1, 2], [2, 3], [3, 1]], "B": [[6, 5], [7, 7], [8, 6]]}
# example = [5, 7]
# knn = KNearestNeighbor()
# print(knn.predict(dataset, example, k=3))
