# -------------------------------------------------------------------------
# AUTHOR: Matthew Le
# FILENAME: svm.py
# SPECIFICATION: This program iterates through all of the hyperparameter combinations to determine the best performing combination.
# FOR: CS 4200- Assignment #3
# TIME SPENT: 45 minutes (approx.)
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import svm
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0
highestAccuracyString = ""

# reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
        X_training.append(row[:-1])
        Y_training.append(row[-1])

# reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        dbTest.append(row)

# created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
# --> add your Python code here

for i in c:  # iterates over c
    for j in degree:  # iterates over degree
        for k in kernel:  # iterates kernel
            for l in decision_function_shape:  # iterates over decision_function_shape

                # Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=i, degree=j, kernel=k, decision_function_shape=l)

                # Fit Random Forest to the training data
                clf.fit(X_training, Y_training)

                # make the classifier prediction for each test sample and start computing its accuracy
                # --> add your Python code here
                total = 0
                correct = 0
                for testSample in dbTest:
                    class_predicted = clf.predict([testSample[:-1]])[0]
                    if int(class_predicted) == int(testSample[-1]):
                        correct += 1
                    total += 1

                # check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                # --> add your Python code here
                accuracy = correct / total
                if accuracy > highestAccuracy:
                    highestAccuracy = accuracy
                    highestAccuracyString = str(highestAccuracy) + ", Parameters: a=" + str(i) + ", degree=" + str(
                        j) + ", kernel= " + str(k) + ", decision_function_shape = '" + str(l) + "'"
                    print("Highest SVM accuracy so far: ",
                          highestAccuracyString)
# print the final, highest accuracy found together with the SVM hyperparameters
#Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
# --> add your Python code here
print("")
print("Highest SVM accuracy: ", highestAccuracyString)
