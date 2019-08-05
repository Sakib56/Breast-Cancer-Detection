import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib import style

dataset = {'A': [[1, 2], [2, 3], [3, 1]], 'B': [[6, 5], [7, 7], [8, 6]]}
example = [5, 7]

def knn(datum, sample, k=3):
    distances = []
    for group in datum:
        for features in datum[group]:
            l2norm = np.linalg.norm(np.array(features)-np.array(sample))
            distances.append([l2norm, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    verdict = Counter(votes).most_common(1)[0][0]

    return verdict 

result = knn(dataset, example, k=3)
print(result)