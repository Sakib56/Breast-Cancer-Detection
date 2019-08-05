import numpy as np
from collections import Counter

class knn():
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    
    def predict(self, datum, sample, k=3):
        distances = []
        for group in datum:
            for features in datum[group]:
                l2norm = np.linalg.norm(np.array(features)-np.array(sample))
                distances.append([l2norm, group])

        votes = [i[1] for i in sorted(distances)[:k]]
        verdict = Counter(votes).most_common(1)[0][0]

        return verdict 

dataset = {'A': [[1, 2], [2, 3], [3, 1]], 'B': [[6, 5], [7, 7], [8, 6]]}
example = [5, 7]
knn = knn()
print(knn.predict(dataset, example, k=3))