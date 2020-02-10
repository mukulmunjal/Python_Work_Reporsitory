# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:39:13 2019

@author: I341052
"""
#Sets-> It is collection of distinct objects
#They Do not hold duplictate valiues
#Stores elements in no perticular order

#Automatically removes the duplicate ITEM and returns set of unique elements.
age = {51,22,33,5,5}
age

emp_name = {'Ram','Joy','John','Josha'}
emp_name

emp_name.add('Mohit')
emp_name

emp_name.add('Joy')
emp_name

emp_name.discard('Joy')

#Clear -> Clears all the names

emp_name.clear()
emp_name

#OPERATIONS on SETS => UNION, INTERSECTION
set_1 = {'A','B','C','E'}
set_2 = {'K','F','A','C'}

#UNION
U = set_1.union(set_2)
#INTERSECTION
I = set_1.intersection(set_2)
#DIFFERNCE
D = set_1.difference(set_2)
#Symmetric Differnece => Returns the elements not common to both sets
SD = set_1.symmetric_difference(set_2)

U
I
D
SD