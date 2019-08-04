import numpy as np

def loadData():
    with open("data.csv") as data:
        dataMat = []
        for i, row in enumerate(data):
            cleanedRow = list(filter(None, map(lambda x: x.strip(), row.split(","))))

            diagosis = cleanedRow[1]
            if diagosis == "M":
                cleanedRow[1] = 1
            if diagosis == "B":
                cleanedRow[1] = 0
            dataMat.append(cleanedRow)

        return np.matrix(dataMat)


### MAIN ###
cancerData = loadData()
print(cancerData.shape)
