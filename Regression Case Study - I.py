# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:49:35 2019

@author: I341052
"""

#Problem Statement - Strok Motors is an e-comm company who act intermediatery b/w parties intersted in selling and 
#buying pre-owned cars.
#They have good amount of data and they want to develop an algorithm to predict the price of the case based on
#various attributes associated with the car.

import pandas as pd
import numpy as np
import seaborn as sb 
import datetime

# ===============================
# Setting dimensions for the plot
# ===============================
size = (11.7, 8.27)
sb.set(rc= {'figure.figsize': (11.7, 8.27)})

#Reading a File
data = pd.read_csv('C:\\cars_sampled.csv')
data

#Copying the data    
data1 = data.copy()

#Structure of the data set
data1.info()

#Summerizing data

data1.describe()
pd.set_option('display.float.format', lambda x: '%.2f' % x)
data1.describe()

#To display the maximum set of columns
pd.set_option('display.max_columns',500)
data1.describe()
data1

data1.columns

#Dropping the unwanted columns 
cols = ['name', 'dateCrawled', 'dateCreated', 'postalCode', 'lastSeen']
data1 = data1.drop(columns = cols, axis = 1)

#====================================================
# Removing duplicate records
data1.drop_duplicates(keep = 'first', inplace = True)    
#470 Duplicate records deleted

#===================================================
#Data Cleaning
#==================================================

#No. of missing values in each column
data1.isnull().sum()

#Variable yearofregistration
cars_distribution_year_wise = data1.yearOfRegistration.value_counts().sort_index()
cars_distribution_year_wise
sum(data1.yearOfRegistration > 2018)
sum(data1.yearOfRegistration < 1950)
sb.regplot(x= data1.yearOfRegistration, y=data1.price, scatter=True, fit_reg=True)

#Working range 1950 and 2018

#Variable PRICE
cars_count = data1.price.value_counts().sort_index()
cars_count
#Creating Histogram
sb.distplot(data1.price)
data1.price.describe()
sb.boxplot(y = data1.price)
sum(data1.price > 150000)
sum(data1.yearOfRegistration < 100)


#Variable PowerPS
power_count = data1.powerPS.value_counts().sort_index()
power_count
#Creating Histogram
sb.distplot(data1.powerPS)
power_count.describe()
sb.boxplot(y = data1.powerPS)
sb.regplot(x= data1.powerPS, y=data1.price, scatter=True, fit_reg=True)
sum(data1.powerPS > 500)
sum(data1.powerPS < 10)

#Working range 10 and 500 => We ACHIEVED THIS BY TRAIL AND ERROR KEEPING 
#IN MIND TO NOT TO LOSE(EXCLUDE) TOO MANY DATA

#Working range of DATA

data1 = data1[
        (data1.yearOfRegistration <=2018)
    &   (data1.yearOfRegistration >=1950)
    &   (data1.price <=150000)
    &   (data1. price >=100)
    &   (data1.powerPS >= 10)
    &   (data1.powerPS <= 500)]
    
#~6700 records are dropped

#To determine the car's age
# In order to do so we need to add the yearOfRegistration + MonthOfRegistration
 # -Converting MonthOfRegistration to year first
 
data1.monthOfRegistration = data1.monthOfRegistration/12

now = datetime.datetime.now()
now
now.year


data1['age_of_car'] = (now.year - data1.yearOfRegistration + data1.monthOfRegistration)
data1.age_of_car
data1.age_of_car.describe()
#Now you see the describe that the variation is not too much b/w mean and median(50%)
#which is good

#Dropping the yearOfRegistration and MonthOfRegistration columns
col = ['yearOfRegistration','monthOfRegistration']
data1 = data1.drop(col, axis = 1)    


#VISUALIZING PARAMETERES

#Age 
sb.distplot(data1.age_of_car)
sb.boxplot( y = data1.age_of_car)

#Price
sb.distplot(data1.price)
sb.boxplot( y = data1.price)

#PowerPS
sb.distplot(data1.powerPS)
sb.boxplot( y = data1.powerPS)

#Visualizing parameters after narrowing working range

#Age_of_car vs Price
sb.regplot(x=data1.age_of_car, y=data1.price, scatter=True, fit_reg=False)
#Inferance: Cars priced higher are newer
#With increase in age price decreases
#However, some cars are priced higher with increase in age

#PowerPS vs Price
sb.regplot(x=data1.powerPS, y=data1.price, scatter=True, fit_reg=False)
#Inferance: With increase in power Price increases


#Comparision with catagorial variables

#Variable - Seller
data1.seller.value_counts()
pd.crosstab(data1.seller, columns = 'count', normalize = True)
sb.countplot(data1.seller)
#Inferance: #Very Very Fewer cars have 'COMMERCIAL' seller ==> hence, insignificant

#Variable - Offertype
data1.offerType.value_counts()
sb.countplot(data1.offerType)
#Inferance: #All case have OFFER as offer_type => Insignificant

#Variable - ABTEST
data1.abtest.value_counts()
pd.crosstab(data1.abtest, columns = 'count', normalize = True)
sb.countplot(data1.abtest)
#Equally Distributed
sb.boxplot(x = data1.abtest, y = data1.price)
#Inferance: #For every price value there is almost 50-50 distribution
#Doesn't affect price => Insignificant 


#Variable - vehicle Type
data1.vehicleType.value_counts()
pd.crosstab(data1.vehicleType, columns = 'count', normalize = True)
sb.countplot(data1.vehicleType)
sb.boxplot(x = data1.vehicleType, y = data1.price)
#Inferance: # 8 types(Limousine, small car, station wagon) are the max. frequency
#Vechicle type affects Price

#Variable - vehicle Type
data1.gearbox.value_counts()
pd.crosstab(data1.gearbox, columns = 'count', normalize = True)
sb.countplot(data1.gearbox)
sb.boxplot(x = data1.gearbox, y = data1.price)
#Inferance: #gearbox type affects Price => Since Manual is priced lower than automatic

#Variable - Model
data1.model.value_counts()
pd.crosstab(data1.model, columns = 'count', normalize = True)
sb.countplot(data1.model)
sb.boxplot(x = data1.model, y = data1.price)
#Inferance: #Cars are distributed over many models
#considered in modelling

#Variable - in Kilometer
data1.kilometer.value_counts().sort_index()
pd.crosstab(data1.kilometer, columns = 'count', normalize = True)
sb.countplot(data1.kilometer)
sb.boxplot(x = data1.kilometer, y = data1.price)
data1.kilometer.describe()
sb.distplot(data1.kilometer, bins=8, kde = False)
#Inferance: #considered in modelling

#Variable - FuelType
data1.fuelType.value_counts()
pd.crosstab(data1.fuelType, columns = 'count', normalize = True)
sb.countplot(data1.fuelType)
sb.boxplot(x = data1.fuelType, y = data1.price)
#Inferance: Affects price as cars price is ranging due to differnet types of fuel types
#Considered for modelling

#Variable - Brand
data1.brand.value_counts()
pd.crosstab(data1.brand, columns = 'count', normalize = True)
sb.countplot(data1.brand)
sb.boxplot(x = data1.brand, y = data1.price)
#Inferance: Cars is being distributed over many brands, hence #Considered for modelling

#Variable - notRepairedDamage
#yes- Car is damaged but not rectified
#No - Car was damaged but was rectified
data1.notRepairedDamage.value_counts()
pd.crosstab(data1.notRepairedDamage, columns = 'count', normalize = True)
sb.countplot(data1.notRepairedDamage)
sb.boxplot(x = data1.notRepairedDamage, y = data1.price)
#Inferance: Cars  that requires damages to be repaired have lower Price


#============================================
#Hence, Now removing insignificent variables
#============================================
col = ['abtest','offerType','seller']
data1 = data1.drop(col, axis =1)
cars_data = data1.copy()

#============================================
# Correlation
#============================================
cars_data.info()
cars_data1 = cars_data.select_dtypes(exclude = [object])
cars_data1
correlation = cars_data1.corr()
correlation

correlation[:, correlation.price].abs().sort_values
#abs = Absolute Values

#============================================
# Now we are going to build a linear regression and Random Forest model on 2 data sets.
# 1. Data obtained by omitting (removing) rows with any missing value.
# 2. Data obtained by imputing the missing values.
#============================================

# 1. Data obtained by omitting (removing) rows with any missing value.

data_omit = cars_data.dropna(axis =0)
#Here Axis=0 menas that to drop any row that contains any number of missing value(Be it 1 cell missing 
#or all cells missing)
data_omit

# Values under catagorial variables are not in numbers(which none of Machine Learning Algo's can interpret)
# So we will dummy encode these columns and genetrate new columns out of it
#For instance - Fuel Type has 4 types - Petrol, Diesal, CNG, LPG. So we will generate 4-1 = 3 columns MORE out 
#of this.

#1 Column have fuel_type = Diesal, other column will be fuel_type = Petrol and so on.
#Now the value under each column will be 0 or 1
# 0 => If it's not present
# 1 => If it's present

#Converting Catagorial values to Dummy variables:
data_omit_dummies = pd.get_dummies(data_omit, drop_first = True)
# drop_first = True means => 2 Columns have same information because the original column could 
# assume a binary value. Its a smart way to keep only one of the 2 columns.

# ===========================================================
# Importing neccessary lib's
# ===========================================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# ===========================================================
# Building model with omitted data
# ===========================================================

# Seperating input and output features
x1 = data_omit_dummies.drop('price', axis = 'columns', inplace = False)
y1 = data_omit_dummies.price
 Before y1 value and after log(y1) value

#We have taken natural LOG since it gives a good BELL shaped graph
# Normal before graph is skewed one.
prices = pd.DataFrame({'1. Before':y1, '2. After ':np.log(y1)})
prices.hist()

# Plotting the variable price - By creating a new varibale to convert the y1 as dataframe
#in DF =>
#Hence transforming the price as log of y1

y1 = np.log(y1)

#Splitting the data to test and train
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.3, random_state = 3)
# TEST Size => means for training it goes 70% and for test it goes 30%
# Random_state = If you don't mention the random_state in the code,
# then whenever you execute your code a new random value is generated
# and the train and test datasets would have different values each time.
# However, if you use a particular value for random_state(random_state = 1 or any other value)
# everytime the result will be same,i.e, same values in train and test datasets.

# X = repersents input features
# Y = repersents ouput features

x_train.shape, x_test.shape, y_train.shape, y_test.shape

# ===========================================================
# Baseline model for omitted data
# ===========================================================
 
"""
We are making a base model to using the TEST data's MEAN value
This is to set the benchmark and to compare with our regression model
"""
# Finding the mean value of test data.
base_pred = np.mean(y_test)
base_pred

base_pred = np.repeat(base_pred, len(y_test))

# Finding RMSE - Root mean squared error
# It's Sq. root of mean squared error
# mean squared error => Diff. b/w test value and predicted value => THEn Square's them => Then,
# Divide by the number of samples.

root_mean_squared_error = np.sqrt(mean_squared_error(y_test, base_pred))
root_mean_squared_error

"""
This is the benchmark for comparision.
Any model we build in future should give the RMSE less than this. That is the objective for US.
"""

# ====================================
# Linear Regression with omitted data
# ====================================

lgr = LinearRegression(fit_intercept = True)

# Model
linear_reg_model = lgr.fit(x_train, y_train)

# Predicting my model on test set
cars_prediction_by_linear_reg_model = linear_reg_model.predict(x_test)
cars_prediction_by_linear_reg_model

# Computing mean squared error and Root mean squared error
lin_reg_root_mean_squared_error = np.sqrt(mean_squared_error(y_test, cars_prediction_by_linear_reg_model))
lin_reg_root_mean_squared_error

# R Squared value => it gives the idea how good my MODEL in explaining the variability in Y
R2_lin_test = linear_reg_model.score(x_test, y_test)
R2_lin_train = linear_reg_model.score(x_train, y_train)
R2_lin_test, R2_lin_train

"""
Values = (0.7658590513951778, 0.7800943845985095) tells that model is good.
The variability what was capture in TRAIN data it's able to capture the same amount of 
Variability(if not more) in test data itself
"""

#Regression Diagnostics - Residual plot analysis => it's the diff b/w the actual and predicted value.
residuals = y_test - cars_prediction_by_linear_reg_model
residuals.describe() # tells the means is 0.00 which is V.good that residuals are ~zero.
#This means that predicted and actual values are closer now.

# =================== 
# Random Forest Model with omitted data
# ===================

# Model Parameters
rf = RandomForestRegressor(n_estimators = 100, max_features = "auto", max_depth = 100, min_samples_split = 2, 
                           min_samples_leaf = 4, random_state = 1)

# Model
model_RF = rf.fit(x_train, y_train)

# Predicting my model on test set
cars_prediction_by_rf_model = rf.predict(x_test)


# Computing mean squared error and Root mean squared error
rf_reg_root_mean_squared_error = np.sqrt(mean_squared_error(y_test, cars_prediction_by_rf_model))
rf_reg_root_mean_squared_error
# the RMSE has come down further which is good.

# R Squared value => it gives the idea how good my MODEL in explaining the variability in Y
R2_rf_test = model_RF.score(x_test, y_test)
R2_rf_train = model_RF.score(x_train, y_train)
R2_rf_test, R2_rf_train


# ================================
# Model building with IMPUTED DATA
# ================================
cars_data.isna().sum()
cars_imputed = cars_data.apply(lambda x: x.fillna(x.median())\
                               if x.dtype == 'float' else\
                               x.fillna(x.value_counts().index[0]))

cars_imputed.isna().sum()

# ================================
# Converting the catagorial variables to dummy variables 
# ================================
cars_imputed = pd.get_dummies(cars_imputed, drop_first = True)

# ====================================
# Linear Regression with IMPUTED data
# ====================================

x2 = cars_imputed.drop('price', axis = 'columns', inplace = False)
y2 = cars_imputed.price

#Before y1 value and after log(y1) value

#We have taken natural LOG since it gives a good BELL shaped graph
# Normal before graph is skewed one.
prices_imputed_val = pd.DataFrame({'1. Before':y2, '2. After ':np.log(y2)})
prices_imputed_val.hist()

# Plotting the variable price - By creating a new varibale to convert the y2 as dataframe
#in DF =>
#Hence transforming the price as log of y2

y2 = np.log(y2)

#Splitting the data to test and train
x_train1, x_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size = 0.3, random_state = 3)
# TEST Size => means for training it goes 70% and for test it goes 30%
# Random_state = If you don't mention the random_state in the code,
# then whenever you execute your code a new random value is generated
# and the train and test datasets would have different values each time.
# However, if you use a particular value for random_state(random_state = 1 or any other value)
# everytime the result will be same,i.e, same values in train and test datasets.

# X = repersents input features
# Y = repersents ouput features

x_train1.shape, x_test1.shape, y_train1.shape, y_test1.shape

# ===========================================================
# Baseline model for imputed data
# ===========================================================
 """
We are making a base model to using the TEST data's MEAN value
This is to set the benchmark and to compare with our regression model
"""
# Finding the mean value of test data.
base_pred1 = np.mean(y_test1)
base_pred1
base_pred1 = np.repeat(base_pred1, len(y_test1))

# Finding RMSE - Root mean squared error
# It's Sq. root of mean squared error
# mean squared error => Diff. b/w test value and predicted value => THEn Square's them => Then,
# Divide by the number of samples.

root_mean_squared_error1 = np.sqrt(mean_squared_error(y_test1, base_pred1))
root_mean_squared_error1

"""
This is the benchmark for comparision.
Any model we build in future should give the RMSE less than this. That is the objective for US.
"""

# ====================================
# Linear Regression with imputed data
# ====================================

lgr2 = LinearRegression(fit_intercept = True)

# Model
linear_reg_model1 = lgr2.fit(x_train1, y_train1)

# Predicting my model on test set
cars_prediction_by_linear_reg_model1 = linear_reg_model1.predict(x_test1)
cars_prediction_by_linear_reg_model1

# Computing mean squared error and Root mean squared error
lin_reg_root_mean_squared_error1 = np.sqrt(mean_squared_error(y_test1, cars_prediction_by_linear_reg_model1))
lin_reg_root_mean_squared_error1

# R Squared value => it gives the idea how good my MODEL in explaining the variability in Y
R2_lin_test1 = linear_reg_model1.score(x_test1, y_test1)
R2_lin_train1 = linear_reg_model1.score(x_train1, y_train1)
R2_lin_test1, R2_lin_train1

"""
Values = (0.702334276621733, 0.7071657774822305) tells that model is good.
The variability what was capture in TRAIN data it's able to capture the same amount of 
Variability(if not more) in test data itself
"""

#Regression Diagnostics - Residual plot analysis => it's the diff b/w the actual and predicted value.
residuals1 = y_test1 - cars_prediction_by_linear_reg_model1
residuals1.describe()

# =================== 
# Random Forest Model with IMPUTED data
# ===================

# Model Parameters
rf1 = RandomForestRegressor(n_estimators = 100, max_features = "auto", max_depth = 100, min_samples_split = 2, 
                           min_samples_leaf = 4, random_state = 1)

# Model
model_RF1 = rf1.fit(x_train1, y_train1)

# Predicting my model on test set
cars_prediction_by_rf_model1 = rf1.predict(x_test1)


# Computing mean squared error and Root mean squared error
rf_reg_root_mean_squared_error1 = np.sqrt(mean_squared_error(y_test1, cars_prediction_by_rf_model1))
rf_reg_root_mean_squared_error1
# the RMSE has come down further which is good.

# R Squared value => it gives the idea how good my MODEL in explaining the variability in Y
R2_rf_test1 = model_RF1.score(x_test1, y_test1)
R2_rf_train1 = model_RF1.score(x_train1, y_train1)
R2_rf_test1, R2_rf_train1



 # ================================================================================================
 # Final Values 
 # ================================================================================================
 print("Metrices for models build from data where missing values were OMITTED")
 print("R Squared value for train from Linear regression = %s"% R2_lin_train)
 print("R Squared value for test from Linear regression =  %s"% R2_lin_test)
 print("R Squared value for train from Random Forest =  %s"%R2_rf_train)
 print("R Squared value for test from Random Forest =  %s"%R2_rf_test)
 print("BASE RMSE of model build from data where missing values were  OMITTED=  %s"%root_mean_squared_error)
 print("RMSE value for test from Linear Regression = %s" %lin_reg_root_mean_squared_error)
 print("RMSE value for test from Random Forest = %s" %rf_reg_root_mean_squared_error)
 print("\n\n")
 
 print("Metrices for models build from data where missing values were IMPUTED")
 print("R Squared value for train from Linear regression = %s "% R2_lin_train1)
 print("R Squared value for test from Linear regression =  %s"% R2_lin_test1)
 print("R Squared value for train from Random Forest =  %s"%R2_rf_train1)
 print("R Squared value for test from Random Forest =  %s"%R2_rf_test1)
 print("BASE RMSE of model build from data where missing values were IMPUTED = %s"%root_mean_squared_error1)
 print("RMSE value for test from Linear Regression = %s" %lin_reg_root_mean_squared_error1)
 print("RMSE value for test from Random Forest = %s" %rf_reg_root_mean_squared_error1)
 print("\n\n")
 
  # ================================================================================================
 # End Of Script
 # ================================================================================================