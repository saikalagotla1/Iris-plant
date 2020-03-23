import numpy as np
from sklearn.svm import SVC
import readData

features = np.array(readData.readcsv(""))
labels = np.array(readData.readcsv(""))

clf = SVC(gamma='auto')
clf.fit(x, y)
SVC(gamma='auto')
print(clf.predict([[-0.8, -1]]))