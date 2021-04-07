# -------------------------------------------------------------------------
# AUTHOR: Matthew Le
# FILENAME: bagging_random_forest.py
# SPECIFICATION: description of the program
# FOR: CS 4200- Assignment #3
# TIME SPENT: Start- 7:40
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
classVotes = []  # this array will be used to count the votes of each classifier

# reading the training data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
        dbTraining.append(row)

# reading the test data in a csv file
with open('optdigits.tes', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        dbTest.append(row)
        # inititalizing the class votes for each test sample
        classVotes.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    print("Started my base and ensemble classifier ...")

    for k in range(20):  # we will create 20 bootstrap samples here (k = 20). One classifier will be created for each bootstrap sample

        bootstrapSample = resample(
            dbTraining, n_samples=len(dbTraining), replace=True)

        # populate the values of X_training and Y_training by using the bootstrapSample
        # --> add your Python code here
        X_training = []
        Y_training = []
        for sample in bootstrapSample:  # for all samples in bootstrapSample
            bootstrapSampleX = []
            # add all values to bootstrapSampleX (then to X_training) except last column
            for i in range(len(sample) - 2):
                bootstrapSampleX.append(sample[i])

            X_training.append(bootstrapSampleX)

            # add last value to Y_training
            Y_training.append(sample[len(sample) - 1])

        # fitting the decision tree to the data
        # we will use a single decision tree without pruning it
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=None)
        clf = clf.fit(X_training, Y_training)

        accuracy = 0

        total = 0
        correct = 0
        for i, testSample in enumerate(dbTest):
            # make the classifier prediction for each test sample and update the corresponding index value in classVotes. For instance,
            # if your first base classifier predicted 2 for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0] will change to classVotes[0,0,1,0,0,0,0,0,0,0].
            # Later, if your second base classifier predicted 3 for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0] will change to classVotes[0,0,1,1,0,0,0,0,0,0]
            # Later, if your third base classifier predicted 3 for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0] will change to classVotes[0,0,1,2,0,0,0,0,0,0]
            # this arrays will consolidate the votes of all classifier for all test samples
            # --> add your Python code here
            # class_predicted = clf.predict([testSample])[0]
            # print("len(X_training): " + str(len(X_training[0])))
            # print("len(Y_training): " + str(len(Y_training[0])))
            # print("testSample: " + str(len(testSample)))
            testSampleData = []
            for j in range(len(testSample) - 2):
                testSampleData.append(testSample[j])
            class_predicted = clf.predict([testSampleData])[0]

            # print("class_predicted: " + str(class_predicted))
            classVotes[int(i)][int(class_predicted)] += 1
            # print("classVotes[i][class_predicted]: " + str(classVotes[i][int(class_predicted)]))

            if k == 0:  # for only the first base classifier, compare the prediction with the true label of the test sample here to start calculating its accuracy
               # --> add your Python code here
                if class_predicted == testSample[len(testSample) - 1]:
                    correct += 1
                total += 1

        if k == 0:  # for only the first base classifier, print its accuracy here
            # --> add your Python code here
            accuracy = correct / total
            print("Finished my base classifier (fast but relatively low accuracy) ...")
            print("My base classifier accuracy: " + str(accuracy))
            print("")

    # now, compare the final ensemble prediction (majority vote in classVotes) for each test sample with the ground truth label to calculate the accuracy of the ensemble classifier (all base classifiers together)
    # --> add your Python code here
    finalTotal = 0
    finalCorrect = 0

    for i, testSample in enumerate(dbTest):
        majority = -1
        majorityVotes = 0
        for j in range(len(classVotes[i]) - 1):
            if classVotes[i][j] > majorityVotes:
                majority = j
                majorityVotes = classVotes[i][j]
        
        #testing
        # print("majority: " + str(majority))
        # print("testSample[len(testSample) - 1]: " + str(testSample[len(testSample) - 1]))

        if int(testSample[len(testSample) - 1]) == int(majority):
            finalCorrect += 1
            # print("correct!!")

        finalTotal += 1 

    # for x in range(len(classVotes) - 1):
    #     majority = -1
    #     majorityVotes = 0
    #     for i in range(len(classVotes[x]) - 1):
    #         if classVotes[x][i] > majorityVotes:
    #             majority = i
    #             majorityVotes = classVotes[x][i]

    #     if dbTest[x][len(dbTest[x]) - 1] == majority:
    #         finalCorrect += 1
    #         print("correct")

    #     finalTotal += 1

    finalAccuracy = finalCorrect / finalTotal
    # printing the ensemble accuracy here
    print("Finished my ensemble classifier (slow but higher accuracy) ...")
    print("My ensemble accuracy: " + str(finalAccuracy))
    print("")

    print("Started Random Forest algorithm ...")

    # Create a Random Forest Classifier
    # this is the number of decision trees that will be generated by Random Forest. The sample of the ensemble method used before
    clf = RandomForestClassifier(n_estimators=20)

    # Fit Random Forest to the training data
    clf.fit(X_training, Y_training)

    # make the Random Forest prediction for each test sample. Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
    # --> add your Python code here

    # compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
    # --> add your Python code here

    # printing Random Forest accuracy here
    print("Random Forest accuracy: " + str(accuracy))

    print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
