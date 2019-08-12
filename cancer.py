import knn
from random import shuffle

# loads data from data.csv into the correct format (dictionary) for my knn
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

# shuffles & splits data into two smaller datasets for training & testing
def crossValidate(data, trainSize=0.8):
    shuffle(data["B"])
    shuffle(data["M"])

    trainingData = {"B": [], "M": []}
    testingData = {"B": [], "M": []}

    for diagnosis in ["M", "B"]:
        size = int(len(data[diagnosis])*trainSize)
        trainingData[diagnosis] = data[diagnosis][:size]
        testingData[diagnosis] = data[diagnosis][size:]

    return trainingData, testingData

# uses a brute force approach to find the best hyperparamters for k and trainSize
def bruteForceBestHyperParams(diagnosis):
    hyperParams = []
    for sizes in [s*0.1 for s in range(4, 9)]:
        for k in [k for k in range(1, 15)]:
            trainingData, testingData = crossValidate(dataset, trainSize=sizes)
            correct = 0
            total = len(testingData[diagnosis])

            for i in range(total):
                example = testingData[diagnosis][i]
                if knn.predict(trainingData, example, k=k) == diagnosis:
                    correct += 1
            hyperParams.append((100*correct/total, sizes, k))

    hyperParams.sort(key=lambda t: t[0])
    return hyperParams


### MAIN ###
if __name__ == "__main__":
    dataset = loadData()
    knn = knn.KNearestNeighbour()

    # from testing best hyperparameters seem to be K = ~7 and trainSize = ~0.6
    trainingData, testingData = crossValidate(dataset, trainSize=0.65)
    k = 7
    score = 100*knn.getScore(trainingData, testingData, k)

    print("average accuracy: {0:.5f}%".format(score))