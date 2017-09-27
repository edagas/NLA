# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 10:24:24 2017

@author: edadagasan
"""
from scipy import *
from pylab import *
import sys
import scipy.linalg as sl
import numpy as np

u=[3,0,0]
v=[0,3,0]

UVt=np.outer(u,v)


U,s,Vh = sl.svd(UVt)

print(s,U,Vh)
