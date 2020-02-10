# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:21:48 2019

@author: I341052

"""
#To work with datafreames
import pandas as pd
import os

#To perform numerical operations
import numpy as np

#To Visualalize the data
import seaborn as sb

#To Partition the data
from sklearn.model_selection import train_test_split

#Importing library for logistic regression
from sklearn.linear_model import LogisticRegression

#Importing performance matrix - accuracy score and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

import pandas as pd
import os
os.chdir('C:\\')
data = pd.read_csv('income(1).csv')
data
data1 = data.copy()
data1
data1.SalStat


#Logistics Regression => It is a machine learning classification algorithm that is used to 
#predict the probability of catagorial dependant variable

#Using Logistics regression we will build classifier model, based on the data

# Reindexing the salary status names to 0,1 
#We are reindexing because machine learning algo's cant work on catagorial data directly
#Hence we are converting it to numbers -> We can assign 0 to Less than 50,000
#1 to greater than 50000

data1.SalStat = data1.SalStat.map({' less than or equal to 50,000':0,' greater than 50,000':1})
data1.SalStat

#Get_Dummies => Convert catagorial variables to num's
new_data = pd.get_dummies(data1, drop_first = True)
new_data

new_data.columns
#Storing the column names as list
column_names = list(new_data.columns)
column_names

#Seperating the input names from LIST of data(to keep INDEPENDANT columns in column list)
feature = list(set(column_names) - set('SalStat'))
feature

#storing the output values in Y 
Y = new_data['SalStat'].values
Y

#Storing the values frpm INPUT features
X = new_data[feature].values
X

#Splitting the data into TRAIN and TEST
#Train x = contains input variable 
#Train y = contains output variable 

#Testx = contains input variable  = Predictions
#Test y = contains output variable  => Actual salary status
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size = 0.3, random_state = 0)

#Make the instance of the model
logistic = LogisticRegression()

#Fillting the values for X and Y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_

#PREDICTION from TEST DATA
prediction = logistic.predict(test_x)
prediction

confusion_matrix = confusion_matrix(test_y, prediction)
confusion_matrix

#Accuracy Score => 
accuracy_score  = accuracy_score(test_y, prediction)
accuracy_score

#Misclassified Samples
print('Misclassfied Samples:%d' %(test_y != prediction).sum())

#===================================================================
# Logistics Regression - Removing insignificant variables
#===================================================================
#reindexing the salary status names to 0,1

data1.SalStat = data1.SalStat.map({' less than or equal to 50,000':0,' greater than 50,000':1})
data1.SalStat

cols = ['gender', 'nativecountry', 'race', 'JobType']
new_data = data1.drop(cols, axis = 1)

new_data = pd.get_dummies(new_data, drop_first = True)
new_data

#STORING THE COLUMN NAMES In variable
columns_list = list(new_data.columns)

#Seperarting the input names from data:
features = list(set(column_names) - set('SalStat'))
feature
 
#storing the output values in Y 
Y = new_data['SalStat'].values
Y


#Storing the values frpm INPUT features
X = new_data[feature].values
X

#Splitting the data into TRAIN and TEST
#Train x = contains input variable 
#Train y = contains output variable 

#Testx = contains input variable  = Predictions
#Test y = contains output variable  => Actual salary status
train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size = 0.3, random_state = 0)

#Make the instance of the model
logistic = LogisticRegression()

#Fillting the values for X and Y
logistic.fit(train_x, train_y)
logistic.coef_
logistic.intercept_

#PREDICTION from TEST DATA
prediction = logistic.predict(test_x)
prediction

#Accuracy Score => 
accuracy_score  = accuracy_score(test_y, prediction)
accuracy_score

# ============================================================================
# KNN
# ============================================================================
 
#Importing the library of KNN

from sklearn.neighbors import KNeighborsClassifier

#import lib for plotting
import matplotlib.pyplot as plt

#Storing KNN classifier(KNN instance) => neighbors = 5
KNN_CLASSIFIER = KNeighborsClassifier(n_neighbors=5)

#Fitting the values for X and Y
KNN_CLASSIFIER.fit(train_x, train_y)

# Predicting the test values with Model
prediction_knn = KNN_CLASSIFIER.predict(test_x)
prediction_knn

#Performance matrix check
confusion_matrix = confusion_matrix(test_y, prediction_knn)
confusion_matrix

accuracy_score = accuracy_score(test_y, prediction_knn)
accuracy_score

#Misclassified Samples
print('Misclassfied Samples:%d' %(test_y != prediction_knn).sum())

misclassified_sample = []
for i in range (1, 20):
    Knn_Classifier = KNeighborsClassifier(n_neighbors=i)
    Knn_Classifier.fit(train_x,train_y)
    prediction_knn_1 = Knn_Classifier.predict(test_x)
    misclassified_sample.append((test_y != prediction_knn).sum())  
    
misclassified_sample
