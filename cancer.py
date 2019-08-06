import knn
import math
from random import shuffle
import numpy as np

def loadData():
    with open("data.csv") as dataCSV:
        data = {"B": [], "M": []}
        for row in dataCSV:
            cleanedRow = list(filter(None, map(lambda x: x.strip(), row.split(","))))
            diagnosis = cleanedRow[1]
            cleanedRow = cleanedRow[2:]

            if diagnosis == "B":
                cleanedRow = list(map(lambda x: float(x), cleanedRow))
                data["B"].append(cleanedRow)
            if diagnosis == "M":
                cleanedRow = list(map(lambda x: float(x), cleanedRow))
                data["M"].append(cleanedRow)

        return data

def crossValidate(data, trainSize=0.8):
    shuffle(data["B"])
    shuffle(data["M"])

    trainingData = {"B": [], "M": []}
    testingData = {"B": [], "M": []}

    for diagnosis in ["M","B"]:
        size = int(len(data[diagnosis])*trainSize)
        trainingData[diagnosis] = data[diagnosis][:size]
        testingData[diagnosis] = data[diagnosis][size:]

    return trainingData, testingData



### MAIN ###
dataset = loadData()
knn = knn.KNearestNeighbor()

trainingData, testingData = crossValidate(dataset, trainSize=0.8)

for k in [k for k in range(1,20,3)]:
    for diagnosis in ["M","B"]:
        correct = 0
        total = len(testingData[diagnosis])
        for i in range(total):
            example = testingData[diagnosis][i]
            if knn.predict(trainingData, example, k=k) == diagnosis:
                correct += 1

        print("K={0}, {1}, accuracy: {2}%".format(k, diagnosis, 100*correct/total))
    print("\n")

