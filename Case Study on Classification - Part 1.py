
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:48:01 2019

@author: I341052
"""
#Problem Statement => Subsidy inc. delivers the subsidy to individuals based on their income.
#Accurate data is one of the hardest piece of data across the world 
#Subsidy inc has a large set of authenticated data on income etc.
#Develop an income classifier system for individuals
#
#Objective is to simplify the the data system by reducing the data variables to be studies w/o sacrificing too much of accuracy
#Such system would help SUBSIDY INC. in planning subsidy, monitoring and preventing misuse.

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

#***
#Importing data
#***
os.chdir('C:\\')
data = pd.read_csv('income(1).csv')
data

#Creating a copy of data
data1 = data.copy()
data1
#Getting to know the data - To check the variables data types
data1.info()

#Check for Missing values
data1.isnull().sum()
#***No missing values!

#Summary of Numerical Variables 
summary_num = data1.describe()
summary_num

summary_cate = data1.describe(include = "O")
summary_cate

data1['JobType'].value_counts()
data1['occupation'].value_counts()

np.unique(data1['JobType'])

data1 = pd.read_csv('income(1).csv', na_values = [" ?"])
data1
data1.isnull().sum()

missing = data1[data1.isnull().any(axis = 1)]
missing
#axis = 1 => to consider at least 1 column value is missing

data2 = data1.dropna(axis = 0)
data2

#Relationship b/w independent variables
correlations = data2.corr()
correlations
#No values are closer to 1; which says they is very week corelations b/w variables.
#***********************************
#So, Variables are not correlated with each other 
#************************************


#Cross tables and Visualization
#Extraction of column names of data 
data2.columns

#Gender Proportion table => using Cross table function 

gender = pd.crosstab(index = data2.gender, columns = 'count', normalize = True)
gender

#Gender vs Salary status:

gender_status = pd.crosstab(index = data2.gender, columns = data2.SalStat, margins = True, normalize = True)
gender_status

#To plot the distribution of Salary Status:
SalStat = sb.countplot(data2.SalStat)


#***************Histogram of Age**********
sb.distplot(data2.age, bins=10, kde = False)
sb.boxplot('SalStat', 'age', data = data2)
data2.groupby('SalStat')['age'].median()

#JobType VS Salary Status  