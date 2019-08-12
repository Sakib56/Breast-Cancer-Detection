# Breast Cancer Detection w/ K-Nearest Neighbours

This is my approach at implementing **KNN from scratch** and applying it to the **Breast Cancer Wisconsin (Diagnostic) dataset**, provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

## How do I use this Repo?
Simple. Just clone the repo into a folder, open a terminal in that directory and run
```python cancer.py```

By default the following code will run...
```python
### MAIN ###
dataset = loadData()
trainingData, testingData = crossValidate(dataset, trainSize=0.65)

k = 7
knn = knn.KNearestNeighbour()
score = 100*knn.getScore(trainingData, testingData, k)

print("average accuracy: {0:.5f}%".format(score))
```
This will:
* load the dataset from data.csv
* cross validate the data, creating a training and testing set
* initialise a new KNN model 
* test KNN on the testing set given the training set
* outputing the average accuracy

Alternatively, you can open a terminal in the directory and just import the relevant modules and run the methods.
For example:
```python
python
Python 3.6.5 (v3.6.5:f59c0932b4, Mar 28 2018, 17:00:18) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
>>> import knn
>>> import cancer
>>>
>>> dataset = cancer.loadData()
>>> trainingData, testingData = cancer.crossValidate(dataset, trainSize=0.65)
>>>
>>> k = 7
>>> knn = knn.KNearestNeighbour()
>>> score = 100*knn.getScore(trainingData, testingData, k)
>>>
>>> print("average accuracy: {0:.5f}%".format(score))
>>>
>>> "accuracy for M diagnosis: 91.33%"
>>> "accuracy for B diagnosis: 96.80%"
>>> "average accuracy: 94.06500%"
```

*Feel free to remove this code and mess around with the methods!*

## cancer.py
#### Loading the Data from .csv
First, I start off by loading the data from the .csv using the **loadData()** function. What is returned will look something like this:

{"B": [b1, b2, ..., bn], "M": [m1, m2, ..., mn]} where bn or mn for any n, is a 30 dimensional list (vector).

###### *Note: it is 30-D instead of 32-D as I have removed the "id" and "diagnosis" columns within loadData()*

for example, loading the data into the format shown above: 
```python
dataset = loadData()
```

#### Cross Validating the Data
As I have a finite dataset to work with, I cannot use all of the dataset to "train" KNN, instead I will cross-validate. This is essentially shuffling the data and splitting into two groups, one for training purposes and another for testing. Think of it like a studying a past paper versus doing a final exam.

The reason this is done is because, if I used all the dataset to train and test my model, it would get a higher accuracy compared to unseen data points (as there will always be one training point exactly on the testing point) ... You wouldn't give your student a final exam which was the same as a past paper.

**crossValidate(data, trainSize=0.8)** is used to do this cross-validation, which returns trainingData and testingData (in that order), where the arguments...

*data* is the training data, a matrix of row vectors which are the datapoints (n by m) and

*trainSize* is the percentage of *data* which is set aside for training (default being 0.8)

for example, creating two datasets for training and testing: 
```python
trainingData, testingData = crossValidate(dataset, trainSize=0.6)
```
let's assume that ```dataset``` had 100 vectors, then ```trainingData``` and ```testingData``` would have 60 and 40 vectors, respectively.
###### *Note: trainingData and testingData are disjoint (share no common elements)*

#### Finding the best hyper-parameters (K and trainSize)
This is a completely inefficient way to find the best hyper parameters for the model. **bruteForceBestHyperParams(diagnosis)** where *diagnosis* is “M” or “B”.

This method will run through values of trainSize from 0.4 to 0.9 for cross validation,

Each time it will also run through values of k from 1 to 15. 

Randomising the dataset and running KNN to create a list of accuracies and hyper parameters.

The basic pseudocode is:
```
BestParameters = []
For trainSizes from 0.4 to 0.9
	For k from 1 to 15
		Randomise dataset
		Get accuracy 
		BestParameters += (accuracy, trainSizes, k)
		
Sort BestParameters by accuracy
Return BestParameters
```

From doing this and collating the results, I have concluded that the best results from my KNN can be achieved by using a 
**trainSize ≈ 0.63** and **k ≈ 7**

## knn.py
#### Predicting a new sample given data
**knn.predict(data, sample, k=n)** is used to predict what class (benign/malignant) *sample*m will be, based on *data* - the other data points around it. where the arguments...

*data* is the training data, a matrix of row vectors which are the datapoints (n by m),

*sample* is a single row vector (1 by m) and

*k* is the number of nearest neighbours the algorithm will consider before classifying *sample*.

for example, predicting what the first value of testingData["M"] (for malignant class) will be: 
```python
knn.predict(trainingData, testingData["M"][0], k=7)
>> "M"
```

#### Finding the Accuracy
**getScore(trainingData, testingData, k)** is used to calculate the accuracy of the model. This is done by inputting *trainingData* and *testingData* and a value for *k*. This method is similar to predict, the difference being that a *testingData* is also a matrix of row vectors instead of a single row vector and a percentage (between 0-1) is returned.

Ideally *trainingData* and *testingData* should be cross validated from the original dataset.

###### *Note: by "accuracy" I mean (number of correct diagnoses)/(total number of cases)*
