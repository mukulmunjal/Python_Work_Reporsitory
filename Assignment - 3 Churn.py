# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:50:42 2019

@author: I341052
"""

import os 
import pandas as pd
import numpy as np
os.chdir("C:\\")
data = pd.read_csv("churn.csv")

data

a = pd.unique(data['customerID'])
a.size
data['customerID'].size

data['TotalCharges'].isnull().sum()

a=data['Dependents'] == "1@#"
a.sum()

data.info()

s = data['tenure'].replace('Four',4)
s
data['tenure']

np.data['tenure'].where('Four',4)

data['tenure'].isnan()

#*****************************************************************************

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
os.chdir("C:\\")
data = pd.read_csv("mtcars.csv",index_col = 0, na_values = ["??","????"])
data.dropna(axis = 0, inplace = True)

plt.hist(data.mpg, color = 'red', edge = 'blue')

plt.scatter(data.mpg, data.wt,color = 'green')

#*************************************************************************
data = pd.read_csv("diamond.csv",index_col = 0, na_values = ["??","????"])
data.dropna(axis = 0, inplace = True)
sb.boxplot(data.cut, data.price)

pd.crosstab(data.cut,columns = 'count',dropna = True)

