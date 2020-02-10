# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:43:15 2019

@author: I341052
"""

import os
import pandas as pd
os.chdir("C:\\")

data = pd.read_csv("Toyota.csv", index_col = 0,na_values = ["??", "????"])
data
#Creation of copy of DATA

data2 = data.copy()
data2

#Frequency Tables
#To check the relationship b/w Variable in data => CATAGORIAL VARIABLES

#Ex: of 1 catagorial 
pd.crosstab(data2['FuelType'],columns = 'count',dropna = True)

#Ex: of 2 catagorial - Two way table => to check the relationship b/w two catagorial variables
pd.crosstab(index = data2.Automatic, columns = data2.FuelType, dropna = True)

#Two - way Tables => Joint Probability
#By Converting the values from Number to Proportion we get JOINT PROBABILITY VALUE => Will be accomplished 
#by paramenter NORMALIZE = True

#Join probablity => It is the likelihood of two INDEPENDENT events happening at the same time.
pd.crosstab(index = data2.Automatic, columns = data2.FuelType, normalize= True, dropna = True)
#HERE value= 0.826347 means that= joint Probability of car having manual gear box(0) when fual type is Petrol.

# TWO WAY TABLE - Marginal Probability=> Probability of occurance of single event

pd.crosstab(index = data2.Automatic, columns = data2.FuelType,margins = True, normalize= True, dropna = True)
#HERE value = 0.945359 means that => Probability of cars having manual gear box(0) when fual type are CNG or Petrol or Diesal

#HERE value = 0.8809885359 means that => Probability of cars having fual type as Petrol and Automatic as 
#manual gear box(0) or Automatic(1)

# TWO WAY TABLE -Conditional Probability => It probabiliy of event A, given that another event B has alreay occured
                #----------------------#
#It is like Given the type of GEAR BOX => Probability of different fual types.
#Set => Normalize = Index
#Row SUM will equal to 1(Equalivant to that ROW) => It is ROW wise probability like for given GEAR TYPE say automatic car the probablity of
# FUEL type is Diesal, CNG or Petrol
pd.crosstab(index = data2.Automatic, columns = data2.FuelType,margins = True, normalize= 'index', dropna = True)

#OR

#Column SUM will equal to 1(Equalivant to that Column) => It is Column wise probability like for given Fuel type say PETROL the probablity of
# GEAR BOX type is Automatic, or manual
##Set => Normalize = Columns
pd.crosstab(index = data2.Automatic, columns = data2.FuelType,margins = True, normalize= 'columns', dropna = True)

#***************************************************CORRELATION**************************************************
#It is the strenght of association b/w 2 variables(mostly NUMERICAL VARIABLES)
#Visual repersentation of correlation -> Scatter plots
#(i) Positive Trend
#(ii) Positive Trend
#(iii) NO correlation or LITTLE
#Corrleation values will be bounded b/w -1 to +1 | 0 means NO CORRELATIONS at all b/w 2 numerical variales.
#Closer to -1 => High -ve correlations 
#Closer to +1 => High +ve correlations (more than 0.7 is fair correlation)
#To check the relationship b/w Variable in data => Numerical VARIABLES
#We need to exclude the CATAGORIAL variables to find the PEARSON's correlation (by default)
#PEARSON's correlation => Used to check the strenght of association b/w 2 NUMERICAL VARIABLES

numerical_data = data2.select_dtypes(exclude=[object])
numerical_data.shape #=> To check the shape 

corr_matrix = numerical_data.corr() #using numerical data as that cotains only NUMBERS 
corr_matrix

#Diagonal values= 1 bcoz price and price correlation will always be 1 (same for others as well)
#-0.878407 => Means -ve correlation b/w age and price of car
#whenever age increases PROCE decreases

#0.581198 => Means +ve correlation b/w weight and price of car 
#whenever weight increases PRICE increases


