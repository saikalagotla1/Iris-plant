import numpy as np
import sys
sys.path.append("../readOrganizeData")
import readData
from sklearn.naive_bayes import GaussianNB

data = readData.readcsv("irisTrainingFeatureList.csv")
labels = readData.readLabels("irisTrainingData.data")
labels = labels.ravel()
data = np.array(data).astype(np.float)

clf = GaussianNB()
clf.fit(data, labels)

testFeatures = readData.readcsv("irisTestFeatures.csv")
testLabels = readData.readcsv("irisTestLabels.csv")

print(clf.predict([[4.5,2.3,1.3,0.3]]))
print(testFeatures)
print(testLabels)

print(clf.score(testFeatures, testLabels))