# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:10:49 2019

@author: I341052
"""

#IF-ELSE-ELIF AND FOR LOOP
#data.insert(4, 'Price Class', "")
#data
top_10_values = data.head(10)
top_10_values
#
#for i in range(0,len(top_10_values['Price']),1):
#        if (top_10_values['Price'][i]<=8450):
#            top_10_values['Price Class'][i]="Low"
#        elif (top_10_values['Price'][i]>11950):
#            top_10_values['Price Class'][i]="High"
#        else:
#            top_10_values['Price Class'][i]="Medium"
#
#

#IF-ELSE-ELIF AND While LOOP

i = 0
while i< len(top_10_values['Price']):
    if (top_10_values['Price'][i]<=8450):
        top_10_values['Price Class'][i]="Low"
    elif (top_10_values['Price'][i]>11950):
        top_10_values['Price Class'][i]="High"
    else:
        top_10_values['Price Class'][i]="Medium"
i = i+1

