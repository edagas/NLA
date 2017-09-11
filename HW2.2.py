# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:10:07 2017

@author: edadagasan
"""

import numpy as np
import scipy.linalg as la


class Orthogonalization():
    
    def __init__(self, givenmatrix):
        self.givenmatrix = givenmatrix
    
    def gramschmidt(self):
        n = len(self.givenmatrix[0])
        m = len(self.givenmatrix)
        trianglematrix = np.zeros(shape=(n,n))
        orthogonalmatrix = np.array(np.zeros(shape=(self.givenmatrix.shape)))
        v = np.zeros(shape=(n,n))
        for j in range(n):
            v[j] = self.givenmatrix[j]
            for i in range(j-1):
                trianglematrix[i][j] = np.dot(orthogonalmatrix[i],self.givenmatrix[j])
                v[j] = v[j] - trianglematrix[i][j]*orthogonalmatrix[i]
            trianglematrix[j][j] = la.norm(v[j])
            orthogonalmatrix[j] = np.divide(v[j],trianglematrix[j][j])

        return orthogonalmatrix
        
    def norm(self, matrix):
        return np.linalg.norm(matrix, ord=2)
        
        
    def qtq(self, matrix):
        return np.dot(matrix.transpose(),matrix)
        
    def deviation(self, matrix):
        qtq = self.qtq(matrix)
        I = np.identity(len(qtq))
        return self.norm(I-qtq)
        
    def allclose(self, matrix):
        qtq = self.qtq(matrix)
        I = np.identity(len(qtq))
        return np.allclose(qtq, I)
        
    def eigenvalues(self, matrix):
        return la.eigvals(self.qtq(matrix))
    
    def determinant(self, matrix):
        return la.det(self.qtq(matrix))

A0 = np.random.rand(6,4)

A = Orthogonalization(A0)
Q = A.gramschmidt()

print(A.norm(Q))
print(A.deviation(Q))
print(A.allclose(Q))
print(A.eigenvalues(Q))
print(A.determinant(Q))

B = la.qr(A0)[0]

print(A.norm(B))
print(A.deviation(B))
print(A.allclose(B))
print(A.eigenvalues(B))
print(A.determinant(B))