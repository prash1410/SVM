from sklearn import svm
import csv
import numpy as np
from sklearn_porter import Porter

file = open("Dataset.csv", "r")
reader = csv.reader(file)

X = []
Y = []
for row in reader:
    varianceArray = np.fromstring(row[0], dtype=np.float, sep=',')
    xZeroCrossingsArray = np.fromstring(row[1], dtype=np.float, sep=',')
    yZeroCrossingsArray = np.fromstring(row[2], dtype=np.float, sep=',')
    zZeroCrossingsArray = np.fromstring(row[3], dtype=np.float, sep=',')
    tempArray = np.array([varianceArray, xZeroCrossingsArray, yZeroCrossingsArray, zZeroCrossingsArray])
    X.append(tempArray)
    Y.append(row[4])

file.close()
X = np.asarray(X)
samples, x, y = X.shape
X = X.reshape((samples, x * y))
print(X.shape)
print(len(Y))

classifier = svm.SVC(kernel='rbf', C=80, gamma=0.25)
classifier.fit(X, Y)
porter = Porter(classifier, language='java')
output = porter.export()


file = open("Prediction.csv", "r")
reader = csv.reader(file)
drivingCounter = 0
for row in reader:
    varianceArray = np.fromstring(row[0], dtype=np.float, sep=',')
    xZeroCrossingsArray = np.fromstring(row[1], dtype=np.float, sep=',')
    yZeroCrossingsArray = np.fromstring(row[2], dtype=np.float, sep=',')
    zZeroCrossingsArray = np.fromstring(row[3], dtype=np.float, sep=',')
    predictionArray = np.array([varianceArray, xZeroCrossingsArray, yZeroCrossingsArray, zZeroCrossingsArray])
    predictionArray = predictionArray.reshape(1, -1)
    print(classifier.predict(predictionArray))


with open('SVC.java', 'w') as f:
    f.write(output)

vectorsString = ""
with open("SVC.java") as f:
    for line in f:
        if "double[][] vectors = " in line:
            vectorsString = line

vectorsString = vectorsString[34:len(vectorsString) - 1]

with open("output.txt", 'w') as f:
    while "}" in vectorsString:
        row = vectorsString[vectorsString.find("{") + 1:vectorsString.find("}")].replace(",", "")
        if row:
            f.write(row + "\n")
        vectorsString = vectorsString[vectorsString.find("}") + 1:len(vectorsString)]

