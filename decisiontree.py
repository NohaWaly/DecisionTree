# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:54:57 2019

@author: nohaw
"""


import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#reading dataset
dataset = read_csv('housevotes.csv', header=None)

#remove unknown by majority
for x in range(1,17):
    if((dataset[x] == 'y').sum()>(dataset[x] == 'n').sum()):
      dataset[x] = dataset[x].replace('?','y')
    else:
      dataset[x] = dataset[x].replace('?','n')  
  
#input & output
X = dataset[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]].values
Y = dataset[0]

#As decision tree can't handle categorical value we'll change values to numeric
for x in range(0,16): 
    numericdata = preprocessing.LabelEncoder()
    numericdata.fit([ 'y', 'n'])
    X[:,x] = numericdata.transform(X[:,x])

ListTestSize = [0.7,0.6,0.5,0.4,0.3]
ListRandState = [10,30,50,70,60]
ListTreeSize = []
ListAccuracy = []
ListTraniningSize = [30,40,50,60,70]
ListMeanAccuracy = []
ListMeanNode = []

for i in ListRandState:
     X_trainset, X_testset, Y_trainset, Y_testset = train_test_split(X, Y, test_size=0.75, random_state=i)
     drugTree = DecisionTreeClassifier(criterion="entropy")
     drugTree.fit(X_trainset,Y_trainset)
     tree = drugTree.tree_
     noOfNodes = tree.node_count
     predTree = drugTree.predict(X_testset)
     accuracy =  metrics.accuracy_score(Y_testset, predTree)
     print("accuracy of decision tress with 25% training set", accuracy)
     print("size of decision tress with 25% training set", noOfNodes)
     print("----------------------------------------------------------------------------")
     
for i in ListTestSize:
    for j in ListRandState: 
        X_trainset, X_testset, Y_trainset, Y_testset = train_test_split(X, Y, test_size=i, random_state=j)
        voteTree = DecisionTreeClassifier(criterion="entropy")
        #Next, we will fit the data with the training feature matrix X_trainset and training  response vector y_trainset 
        voteTree.fit(X_trainset,Y_trainset)
        tree = voteTree.tree_
        noOfNodes = tree.node_count
        #Let's make some predictions on the testing dataset and store it into a variable called predTree.
        predTree = voteTree.predict(X_testset)
        accuracy =  metrics.accuracy_score(Y_testset, predTree)
        ListAccuracy.append(accuracy)
        ListTreeSize.append(noOfNodes)
    print("DecisionTrees's minimum accuracy of training set ",(1-i)*435," is ", min(ListAccuracy))
    print("DecisionTrees's maximum accuracy of training set ",(1-i)*435," is ", max(ListAccuracy))
    print("DecisionTrees's mean accuracy of training set ",(1-i)*435," is ",sum(ListAccuracy)/len(ListAccuracy))
    ListMeanAccuracy.append(sum(ListAccuracy)/len(ListAccuracy))
    print("DecisionTrees's minimum number of nodes of training set ",(1-i)*435," is ", min(ListTreeSize))
    print("DecisionTrees's maximum number of nodes of training set ",(1-i)*435," is ", max(ListTreeSize))
    print("DecisionTrees's mean nodes of training set ",(1-i)*435," is ",sum(ListTreeSize)/len(ListTreeSize))
    ListMeanNode.append(sum(ListTreeSize)/len(ListTreeSize))
    print("----------------------------------------------------------------------------")
    
#plot accuary & training set size
y = np.array(ListMeanAccuracy)
x = np.array(ListTraniningSize)
plt.plot(x,y)
plt.xlabel("Training set size")
plt.ylabel("Accuracy")
plt.show()

#plot number of nodes & training set size
y = np.array(ListMeanNode)
x = np.array(ListTraniningSize)
plt.plot(x,y)
plt.xlabel("Training set size")
plt.ylabel("Number of nodes")
plt.show()







