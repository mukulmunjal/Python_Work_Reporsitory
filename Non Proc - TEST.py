# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:21:30 2019

@author: I341052
"""

import os
import pandas as pd
import numpy as np

os.chdir('C:\\')

data = pd.read_csv('microlending_data.csv')
data_old = pd.read_csv('lendingdata.csv')
data
data.info()
data.dtypes

data.describe()
data.isnull().sum()

Y = data.borrower_genders
Y.isnull().sum()
mean_val = data.borrower_genders.median()[0]
mean_val
Y = Y.fillna(mode_val)
Y.isnull().sum()


data.borrower_genders.isna().sum()
data.dropna(subset = ['borrower_genders'],inplace = True)
data.borrower_genders
data.borrower_genders.isna().sum()

count = data.borrower_genders.value_counts()
count
float(count[0])/float(count.sum())*100
float(count[1])/float(count.sum())*100
float(count[2])/float(count.sum())*100

sector = list(data.sector.values)
sector['Service']
data.loan_amount.max()


len(data.sector.values)
loans = []
for i in range(0,len(data.sector.values),1):
    if (data.sector[i]== 'Services'):
        loan = float(data.loan_amount.values[i])
        loans.append(loan)

max(loans)

data.corr(method ='pearson') 
