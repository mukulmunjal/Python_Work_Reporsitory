# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:46:18 2019

@author: I341052
"""

import pandas as pd
import os

import matplotlib.pyplot as plt
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

#***
#Importing data
#***
os.chdir('C:\\')
data = pd.read_csv('People Charm case.csv')
data
data.isnull().sum()

data.columns
np.unique(data.salary)

a = data.describe()
a

dept_sal = pd.crosstab(index = data.dept, columns = data.salary)
dept_sal

box = sb.boxplot(data.numberOfProjects)

at = sb.boxplot('lastEvaluation', 'numberOfProjects', data = data)
at = data.groupby('lastEvaluation')['numberOfProjects'].median()

sb.his

plt.hist(data.avgMonthlyHours, bins = 10)