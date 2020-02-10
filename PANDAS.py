# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:55:16 2019

@author: I341052
"""

#Provides high performance easy to use data structures amd analysis tool for PYTHON language
#PANDAS means that Panel data => ECONOMATRICS term for multidimensional data

#Pandas deals with Data Frame (2-Dimentional, Size -> Mutable)

import os #=> to change the working directory
import pandas as pd #=>pandas lib to work with DATA FRAMES
import numpy as np #=> To perform the numerical computations on data.

os.chdir("C:\\")
data = pd.read_csv("Toyota.csv",index_col = 0)
data = pd.read_csv("bonds.txt")
data

#Creating copy of original Data
#In Python, there are two ways to create copies
#   o Shallow copy
#   o Deep copy

#Shallow copy => It only creates a new variable that shares the reference of the original object 
#Any changes made to a copy of object will be reflected in the original object as well

shallow_copy_data = data.copy(deep = False)

shallow_copy_data

#Deep copy => In case of deep copy, a copy of object is copied in other object with no reference to the original
#Any changes made to a copy of object will not be reflected in the original object
deep_copy_data = data.copy(deep = True)
deep_copy_data

#Index details
data.index

#Columns details
data.columns
data.size

#Shape of rows and columns
data.shape

data.memory_usage()

#What dimentional data is it?
data.ndim

#6 items from TOP
data.head(6)
#6 items from BOTTOM
data.tail(6)

#To access a scalar value, fastest method is to access it with AT and IAT methods.
#at provides label-based scalar lookups
data.at[98, 'KM']

data
#iat provides integer-based lookups
data.iat[5,0]

data.loc[:,'FuelType']

data.dtypes

#returns a subset of the columns from dataframe based on the column dtypes =>(EXCLUDE or INCLUDE can be done as reuired)
data.select_dtypes(include = [object])

#Concise summery of data frame => INFO()
data.info()

#Conversion of variable's data types
data['Price'] = data['Price'].astype('object')
data['Price']
#OR
data.Price = data.Price.astype(object)
data.dtypes

data['FuelType'].nbytes

data.at[98,'FuelType'].replace('CNG', 'Petrol')
data.at['FuelType'].replace('Petrol',3)

data.isnull().sum()

