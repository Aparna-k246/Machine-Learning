# -*- coding: utf-8 -*-
"""CrossValidation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k9PE8ZUNIUjMF-qONY8vHm8lpKY5izUG
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

iris = datasets.load_iris()
xtrain , xtest , ytrain , ytest = train_test_split(iris.data , iris.target , test_size = 0.2)

clf = LinearRegression()
cross_val_score(clf, iris.data, iris.target, cv = KFold(3, True, 0 ))