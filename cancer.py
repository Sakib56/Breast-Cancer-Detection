import knn
import math
from random import shuffle

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

def bruteForceBestHyperParams(diagnosis):
    hyperParams = []
    for sizes in [s*0.1 for s in range(1,10)]:
        trainingData, testingData = crossValidate(dataset, trainSize=sizes)

        for k in [k for k in range(1,21)]:
            correct = 0
            total = len(testingData[diagnosis])

            for i in range(total):
                example = testingData[diagnosis][i]
                if knn.predict(trainingData, example, k=k) == diagnosis: correct += 1
            hyperParams.append((100*correct/total,sizes,k))
        #         print("K={0}, ans: {1}, trainSize: {2}, accuracy: {3:.4f}%".format(k, diagnosis, sizes, 100*correct/total))
        # print("\n")
    hyperParams.sort(key=lambda tup: tup[0])
    return hyperParams



### MAIN ###
dataset = loadData()
knn = knn.KNearestNeighbor()

for params in bruteForceBestHyperParams("M"):
    print(params)