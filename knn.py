import numpy as np
from collections import Counter


class KNearestNeighbour():
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    # predicts a what class sample belongs to, based on data
    def predict(self, data, sample, k=5):
        # calculate l2norm between sample vector and everyother vector
        distances = [[np.linalg.norm(np.array(features)-np.array(sample)), diagnosis]
                     for diagnosis in data for features in data[diagnosis]]

        # list of distances are then sorted and the top k are considered
        # "vote" is cast by top k and verdict is decided
        votes = [di[1] for di in sorted(distances)[:k]]
        verdict = Counter(votes).most_common(1)[0][0]

        return verdict

    # tests knn using training and testing data and returns a percentage
    def getScore(self, trainingData, testingData, k):
        avgAcc = 0
        for diagnosis in ["M", "B"]:
            correct = 0
            total = len(testingData[diagnosis])
            for i in range(total):
                example = testingData[diagnosis][i]
                correct += 1 if self.predict(trainingData, example, k=k) == diagnosis else 0
            avgAcc += correct/total
            print("accuracy for {0} diagnosis: {1:.2f}%".format(diagnosis, 100*correct/total), end="\n")
        return avgAcc/2
