# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:24:48 2019

@author: I341052
"""

import numpy as np

a_mat = np.matrix(np.arange(1,10)).reshape(3,3)
a_mat

#To Find the determinant of matrix
np.linalg.det(a_mat)

#To Find the rank of matrix
np.linalg.matrix_rank(a_mat)

#To Find the Inverse of matrix
np.linalg.inv(a_mat)

#Consider a system of linear equations
#3x + y + 2z = 2
#3x + 2y + 5z = −1
#6x + 7y + 8z = 3
#Now we can write the equations in the form of Ax=b

A = np.matrix('3,1,2;3,2,5;6,7,8').reshape(3,3)
B= np.matrix('2,-1,3').reshape(3,1)
A
B

#To SOlve eq. Ax=B we can do that by below syntax:
sol_linear_eq = np.linalg.solve(A,B)
sol_linear_eq