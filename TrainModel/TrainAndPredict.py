import numpy as np
import sys
sys.path.append("../readOrganizeData")
import readData
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

data = readData.readcsv("irisTrainingFeatures.csv")
labels = readData.readcsv("NumericalIrisTrainingLabels.csv")
data = np.array(data).astype(np.float)
labels = np.array(labels).astype(np.float)
labels = labels.ravel()

clf = GaussianNB()
clf.fit(data, labels)

testFeatures = readData.readcsv("irisTestFeatures.csv")
testLabels = readData.readcsv("NumericalIrisTestLabels.csv")
testFeatures = np.array(testFeatures).astype(np.float)
testLabels = np.array(testLabels).astype(np.float)
labels = labels.ravel()

print(clf.predict([[4.5,2.3,1.3,0.3]]))

print("Score = ", clf.score(testFeatures, testLabels)*100, "%")
print()