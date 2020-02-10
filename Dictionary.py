# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:38:13 2019

@author: I341052
"""

#Dictionary => Python Dictionary is an example of HASH-VALUE data structures
#It Works like a Key-Value Pairs, where KEY is mapped to VALUES

#Different types of FUELS

Fuel_type = {'Petrol':1, 'Diesal':2, 'CNG':3}

print(Fuel_type)

#To Know the value of Diesal
print(Fuel_type['Diesal'])

#To Access the keys
Fuel_type.keys()

#To Access the values
Fuel_type.values()

#To Access both keys and values
Fuel_type.items()

#Adding a new KEY-VALUE pair
Fuel_type['Electric'] = 5

#Updated the Key-Value pair
Fuel_type.update({'Electric': 4})

Fuel_type

#delete the Key Value pair
del Fuel_type['Petrol']

#Clear => Removes all the Key Value Pairs

Fuel_type.clear()

Fuel_type