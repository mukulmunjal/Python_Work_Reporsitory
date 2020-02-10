# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:59:00 2019

@author: I341052
"""

import os as os
import pandas as pd

os.chdir("C:\\")

#Importing CSV file => 
mtcars = pd.read_csv("mtcars.csv")
mtcars

mtcars['disp'].dtype

mtcars = pd.read_csv("mtcars.csv",index_col = 0)
mtcars

#Importing data from EXCEL file from perticular SHEET in EXCEL file => 
RSD_PREP_WORK = pd.read_excel('Work Monitor - RSD_Task.xlsx', sheet_name = 'PREP')
RSD_PREP_WORK

#Importing TEXT file => 
mtcars = pd.read_table("bonds.txt")
mtcars

mtcars = pd.read_table("bonds.txt", delimiter = '\t')
mtcars
