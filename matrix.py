# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 23:40:00 2019

@author: I341052
"""

import numpy as np

a = np.matrix("1,2,3,5;5,6,7,8;8,10,11,12")
a

a.shape
a.size

new_col = np.matrix("2,1,3")
#Addition in Column Wise
np.insert(a, 0, new_col, axis = 1 )


new_row = np.matrix("9,9,0")
#Addition in Row Wise
a_new = np.insert(a, 0, new_row, axis = 1 )

a_new
a[2,2] = 10

a[1,:]

#Matrix Addition 
mat1 = np.matrix(np.arange(1,10)).reshape(3,3)
mat2 = np.matrix(np.arange(11,20)).reshape(3,3)
np.add(mat1, mat2)
np.subtract(mat1, mat2)
#BElow is the matrix multiplication - how we do on manually
np.dot(mat1, mat2)

np.transpose(mat2)

#multiplication ,of matrix ELEMENT WISE (1*1)
np.multiply(mat1,mat2)
#Division,of matrix ELEMENT WISE (1*1)
np.divide(mat1,mat2)

