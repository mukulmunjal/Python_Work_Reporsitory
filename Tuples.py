# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:13:39 2019

@author: I341052
"""

#Tuples - > Consist an ordered collection of Object
#It is IMMUTABLE -> Once created, they cannot be modified

#Creation of tuple of EMP-JOHN details

John_Details = ('I341002','John', 1000000, 41)
print(John_Details)

print(John_Details[-2])

#Used to access a set of elements from a tuple from a range of index number [x:y]
#x -> Inclusive, Y-> Exclusive
#Elements accessed from x to y-1
print(John_Details[0:3])

#Lenght of Tuple
len(John_Details)
min(John_Details)

#Tuple 2
John_Profile = ('CoE_Consultant','T3', 'L3')

#Tuple 1 + Tuple 2

John_complete_details = (John_Details+John_Profile)
John_complete_details