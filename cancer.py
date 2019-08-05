import numpy as np
import knn

def loadData():
    with open("data.csv") as dataCSV:
        data = []
        targets = []
        for row in dataCSV:
            # removing all nonalphanumeric chars from dataCSV
            cleanedRow = list(filter(None, map(lambda x: x.strip(), row.split(","))))
            data.append(cleanedRow)
            if cleanedRow[1] == "M":
                targets.append([0,1])
            if cleanedRow[1] == "B":
                targets.append([1,0])

        dataMat = np.matrix(data)
        dataTargets = np.matrix(targets)

        dataMat = np.delete(dataMat, 0, axis=0)
        dataTargets = np.delete(dataTargets, 0, axis=0)

        dataMat = np.delete(dataMat, 0, axis=1)
        dataMat = np.delete(dataMat, 0, axis=1)

        return dataMat.astype(np.float), dataTargets.astype(np.int)

### MAIN ###
data, targets = loadData()

Xtrain = data[0:284, :]
Ytrain = targets[0:284, :]
Xtest = data[285:569, :]
Ytest = targets[285:569, :]