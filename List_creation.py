# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:36:01 2019

@author: I341052
"""

id = [1,2,4,5]
name = ["A","B","C","D"]
num_emp = 4

list = [id, name, num_emp]
list
print(list)

#Indexing
print(name)
print(list[0][0])
list[-2:]
#Change or Update the values in LIST
list[1][3] = 'Mukul'
print(list)

#Appending a value in LIST 
sal = [100,2000,30000,29900]
list.append(sal)
print(list)

dept = ["SALES","IT","CoE","IMS"]
list.insert(2,dept)
list

#To delete the item from LIST
del list[3]

#To Remove the value - removes the first matching element
new_list = ["High","Low","High","Low"]
new_list.remove("Low")
new_list

#Display the object that is being removed from the list
new_list.pop(2)
new_list

