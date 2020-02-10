# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:10:04 2019

@author: I341052
"""
# Data Visualizatoin via SEABORN liberary
#It is build on MATPLOTLIB library.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_csv('C:\\Toyota.csv',index_col = 0, na_values = ['??','????'])
data

#To Remove the missing values from DF
data.dropna(axis = 0, inplace = True)
data

#Scattetr Plot
#Below is to give the theme to the plot
sb.set(style="darkgrid")

# To use scatter plot with the help of SEABORN -> We need to use REGPLOT => It basically means 
#REGRESSION PLOT of 2 variables.
sb.regplot(data.Age,data.Price)
#The line coming in GRAPH is due to the fact that by default fit_reg = TRUE
#It bascially estimates and plot a regression model relating X and Y variables.

#Below command is Ex. to show graph without the fit_reg;  fit_reg = False
sb.regplot(data.Age,data.Price,fit_reg=False)

#To customize the appearances of the markers use MARKER = '*' where * = any symbol
#This will be very useful whenever we want to diferntiate w.r.t diffenent
sb.regplot(data.Age,data.Price,fit_reg=False, marker='*')

#Scatter Plot of AGE vs Price by FUELTYPE
#Above canb be achieved with parameter = HUE in LMPLOT
#LMPLOT =>It combines REGPLOT and FacetGrid 
sb.lmplot(x='Age',y='Price',fit_reg=False,data = data,hue='FuelType',legend=True, palette='Set1')

#******************HISTOGRAM by SEABORN*************
#BY default it gives the kernal density estimate, to remove that use => KDE = false
sb.distplot(data.Age,kde = False)

#=> to fix the number of Bins(Which are nothing but range or Interval)
sb.distplot(data.Age,kde = False,bins = 6)

#BAR PLOT => To plot the frequency distibution of any catagorial variable
sb.countplot(x = 'FuelType', data = data)

# Group BAR Plot => Grouped BAR plot of Fuel Type and Gear Mode(Automatic) 
sb.countplot(x='FuelType', hue ='Automatic',data = data)

#Box and Whiskers PLOT for NUMERICAL VARIABLES
#The visual representation of the statistical five number summary of variable(s) is given by Box-and-Whisker plot
#The Five number statistical number includes => Mean, Median, and 3 quantiles(1st,2nd and 3rd)
sb.boxplot(y=data.Price)
#Horizontal Lines are Whiskers
#Horizontal Line at 5000 is called as LOWER WHISKER the repersentation of minimum value, This exclude the OUTLIERS
##Horizontal Line b/w 15000 to 20000 is called as UPPER WHISKER the repersentation of maximum value, This also exclude the OUTLIERS

#The lowest line of box repersents the 1st Quantile(i.e. 25% of the cars prices are less than 8000 approx)
#The Median line of box repersents the 2nd Quantile(AKA MEDIAN)(i.e. 50% of the cars prices are less than 10000 approx)
#The Upper line of box repersents the 3rd Quantile(i.e. 75% of the cars prices are around of 12000 approx)
#The points lying outisde the Whiskers are called as OUTLIERS 

#BOX PLOT for NUMERICAL vs CATAGORIAL variables
#Price of cars for various Fuel Types
sb.boxplot(x=data.FuelType, y = data.Price)

#Price of cars for various Fuel Types and Automatic
sb.boxplot(x=data.FuelType, y = data.Price, hue = data.Automatic) 
#You see that NO AUTOMATIC cars is available for Fuel Type CNG or Diesal and Hence GearBox has come only for Petrol in Green color

#Box and Whiskers PLOT and HISTOGRAM 
#Multiple Plots in a window => 

#Split the plotting into 2 windows(2 parts)
    f,(ax_box, ax_hist) = plt.subplots(2, gridspec_kw = {"height_ratios": (.15, .85)})
    # ax_box repersents the axis for BOXPLOT and ax_hist repersents the axis for HISTOGRAM
    
    sb.boxplot(data.Price, ax = ax_box)
    sb.distplot(data.Price, ax = ax_hist)

#Pairwise Plots => used to plot the pairwise relationship in a dataset.
#Creates scatter plots fot joint relationship and histograms for univariate distributions.
    
sb.pairplot(data, kind = "scatter", hue = 'FuelType')
plt.show()



