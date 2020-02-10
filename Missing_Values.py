# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 17:25:21 2019

@author: I341052
"""

#How to identify the missing values.
import os 
import pandas as pd
import numpy as np

data = pd.read_csv('C:\\Toyota.csv',index_col = 0, na_values = ['??','????'])
data1 = data.copy()
data2 = data1.copy()

#In pandas DF, missing number is repersented by NaN(Not a Number)

# to Check the missing values in data fream
#Identifying the missing Values
# To check the count of missing values in each column
data2.isna().sum()
#OR
data2.isnull().sum()

#Now to check are there any combination of values in same row of missing values, or is it like 
#for a row in which number of age is missing is diifenent with row which has missing value in other column

#Subsetting the rows that have 1 or more missing values
missing = data2[data2.isna().any(axis =1)]
missing.shape

#2 ways to fill the missing values
#(1)=> Fill the values by Mean/Median in case of numerical value
#(2)=> Fill the missing values which has the class of manixmum count, in case of catagorial value

#Now, We should look the description of data whether numerical variable be imputed with MEAN or MEDIAN
#since in some case filling the values by mean is problem because there could be extreme value
#that might cause mean value very high or very low which can mislead the data, hence we should go 
#with Median, because in odd number we would be haing the exactly 
#the middle value after sorting all the values.
#In case of ever number , we will take middle two and do average of them and use that value
#******************************1ST WAY****************************************************
#Kind of(Not always but usually)For  Numerical Values
data2.describe()
#Median is repersented by 50%(2nd Quantile)

#Imputing the missing values of AGE variable
#Calculating MEAN of Age
data.Age.mean()
data.Age.median()

#To Fill NA/NaN values using the specified value
data2.Age.fillna(data.Age.median(),inplace = True)

#******************************2ND WAY****************************************************
#Kind of(Not always but usually) for  Catagorial Values
#Returns a series containaing counts of unique values. 
data2.FuelType.value_counts()
#Most unique value in FuelType is Petrol(as we have checked with INDEX 0(on highest number the value is))
#
data2.FuelType.value_counts().index[0]

#So, as above we have determined the most unique accessed value, now to fill that in NA or NaN value is:
data2.FuelType.fillna(data2.FuelType.value_counts().index[0], inplace =True)

#Checking for missing values after filling values in AGE and FuelType Column
data2.isnull().sum()

#If we have missing values in huge number of Columns like 50 to 60 then those can be filled using LAMBDA in 1 SHOT
#To fill NA/NaN values in both Numerical or Catagorial variables in 1 shot

data3 = data.copy()
data3 = data3.apply(lambda x:x.fillna(x.mean()) \
                    if x.dtype =='float' else \
                    x.fillna(x.value_counts().index[0]))
data3.isna().sum()
