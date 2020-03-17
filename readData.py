import sys
import pandas as pd
import scipy
import matplotlib
import sklearn
import numpy

from sklearn import GaussianNB

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

data = pd.read_csv("irisData.data", usecols=[0,1,2,3])
labels = pd.read_csv("irisData.data", usecols=[4])

print(data)

