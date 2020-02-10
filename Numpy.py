# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:23:23 2019

@author: I341052
"""

#NUMPY => Numerical Python(Numerical Computations in Python)
#Creation of ARRAY with NUMPY-> Ordered collection of basic data type of given Lenght

import numpy as np

x = np.array([2,3,4,5])

print[type(x)]

#Numpy can handle differnt catagorial entities.
#All elements are coerced to same data type
x = np.array([2,3,'n',10])
print(x)

np.linspace(1,10,5,dtype = int, retstep = True)
#retstep => Returns the samples as well as Step value(2.25 in this case - incremented)
np.linspace(1,10,2)

# Equally Steped numbers in given range
np.arange(0,10,2)

#Returns the array of given shape -> Fulled with ONES
np.ones((9,9), int)

#Returns the array of given shape -> Fulled with ZEROS
np.zeros((9,9), int)

#Retunes an array of give shape with random values
np.random.rand(2,2)

#Retunes equally spaced numbers based on log space
#np.logspace(start, stop, num, endpoint = T/F, base, dtype)
np.logspace(1, 10, 5,True, 10.0, int)

#TIMEIT can be used to measure the execution tiem(RUN TIME) for snippets of code

timeit np.logspace(1, 10, 5,True, 10.0, int)

#RESHAPE() => Recast an array to new shape without changing it's data
np.arange(1,10).reshape(3,3)

#Array 1st
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
a
a.shape

#Array 2nd
b = np.arange(11,20).reshape(3,3)
b

#Addition and Multiplication of 2 array's
np.add(a,b)
np.multiply(a,b)

#Accessing values from Array - Index
a[0,1]
a[0:2]

a_subset = a[:2,:2]
a_subset

a[0,0] = 12
a

np.transpose(a)
a.transpose()
# To add the values at the end of array => ROW WISE
np.append(a,[[9,8,5]],axis =0)

# To add the values at the end of array => COLUMN WISE
new_col = np.array([10,6,7]).reshape(3,1)
print(new_col)

np.append(a,new_col,axis= 1)

#INSERT=> Adds values to given position and axis in array

np.insert(a,1,[3,4,1],axis = 0)

a
a_del = np.delete(a,1,axis  = 0)
a_del
