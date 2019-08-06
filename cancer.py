import numpy as np
import knn

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

### MAIN ###
dataset = loadData()
knn = knn.KNearestNeighbor()

diagnosis = "M"

for i in range(len(dataset[diagnosis])):
    example = dataset[diagnosis][i]
    print(knn.predict(dataset, example, k=5))