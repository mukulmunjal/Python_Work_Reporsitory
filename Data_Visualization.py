# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 23:13:39 2019

@author: I341052
"""

#Why Visualize data
#(i) Observe the patterns 
#(ii)Identify extreme values that could be anomalies
#(iii) easy interpretations

#matplotlib => To create 2D graphs and plots
#-Scatter Plot => points that obtained for 2 different variables plotted on horizontal and vertical axes
#Used to convey the relationship b/w 2 numerical variables.
#It is also called correlation plot which tells the correlation of 2 variables.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("C:\\Toyota.csv", index_col = 0, na_values = ["??","????"])

data.dropna(axis = 0, inplace = True)
plt.scatter(data.Age, data.Price, c='blue')
plt.title('Scatter Plot of Age Vs Price of cars')
plt.xlabel('Age(Months)')
plt.ylabel('Price(INR)')
plt.show()

#HISTOGRAM => Graphical repersentation of data using bars of differnent heights
#for Numerical variables
plt.hist(data.KM,color="blue", edgecolor = "red", bins=9)
plt.title('Histogram of KM')
plt.xlabel('KM')
plt.ylabel('Frequency')
plt.show()

#BarPLOT => Categorial data with rectangular bars with lengths to propostion of the count they repersent
#for catagorial variables

#When to Use BAR PLOTS
# To repersents the frequently distribution of catagorial variables
count = [979,120,12]
fueltype = ('Petrol','Diesal','CNG')
index = np.arange(len(fueltype))

plt.bar(index, count, color = ['Red','Blue','cyan'])
plt.title('BAR PLOT of Fuel Type')
plt.xlabel('Fuel Type')
plt.ylabel('Frequency')
plt.xticks(index, fueltype, rotation = 90)
plt.show()
