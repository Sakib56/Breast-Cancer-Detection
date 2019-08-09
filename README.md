# Breast Cancer Diagnosis w/ K-Nearest Neighbours

This is my approach at implementing **KNN from scratch** and applying it to the **Breast Cancer Wisconsin (Diagnostic) dataset**, provided by [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)

## knn.py
This file current contains two methods.
**knn.predict(data, sample, k=n)** where...

*data* is the training data i.e. an n by m matrix of row vectors (which are the datapoints),

*sample* is a single row vector i.e. 1 by m

and *k* is the number of nearest neighbours the algorithm will consider before classifying sample.

## cancer.py
This is where all of the magic happens...

#### Loading the Data from .csv
first I start off by loading the data from the .csv into a dictionary using the **loadData()** function. What is returned will look something like this:

```{"B": [b1, b2, ..., bn], "M": [m1, m2, ..., mn]}``` where bn or mn for any n, is a 30 dimensional list (vector). 

*Note: it is 30D instead of 32D as I have removed the "id" and "diagnosis" columns within loadData()*

#### Cross-Validating the Data
As we have a finite dataset, we can not use all of the dataset to "train" KNN, instead we will cross-validate. This is essentially shuffling the data and splitting into two groups (in my case).

The reason this is done is because, I used all of the dataset to train and test my model, it would get a higher accuracy compared to unseen data points (as there will always be one training point exactly on the testing point)

## data.csv


