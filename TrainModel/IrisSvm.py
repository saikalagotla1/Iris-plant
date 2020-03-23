import sys
import numpy as np
from sklearn.svm import SVC
sys.path.append("../readOrganizeData")
import readData

features = np.array(readData.readcsv("IrisTrainingFeatures.csv"))
labels = np.array(readData.readcsv("NumericalIrisTrainingLabels.csv"))
labels = labels.ravel()

clf = SVC(gamma='auto')
clf.fit(features, labels)
SVC(gamma='auto')
print(clf.predict([[4.5,2.3,1.3,0.3]]))

testFeatures = np.array(readData.readcsv("IrisTestFeatures.csv"))
testLabels = np.array(readData.readcsv("NumericalIrisTestLabels.csv"))
print("Score = ", clf.score(testFeatures, testLabels)*100, "%")