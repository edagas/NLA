# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:02:50 2017

@author: edadagasan
"""
import scipy.linalg as sl
import numpy as np

H = sl.hilbert(50)
H_inv = sl.invhilbert(50)

U,s,V = sl.svd(H)   #U unitary matrix having left sing vectors as columns
                    #s vector with sing values ordered (greatest first)
S = np.diag(s)      #constructing a diag matrix 

b = U[:,0] #first column
delta_b = U[:,len(s)-1] #last column
    
x = H_inv@b
delta_x = H_inv@delta_b


K = (sl.norm(delta_x, ord=2)/sl.norm(x, ord=2))/(sl.norm(delta_b, ord=2)/sl.norm(b, ord=2))
    
K_A = sl.norm(H, ord=2)*sl.norm(H_inv, ord=2)

print(K, K_A)