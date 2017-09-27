# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:10:07 2017

@author: edadagasan
"""

import numpy as np
import scipy.linalg as la


class Orthogonalization():
    """
    A class of matrices with methods that orthogonalize the objects with
    different algorithms and that evaluate the result in different ways.
    """
    
    def __init__(self, givenmatrix):
        self.givenmatrix = givenmatrix
    
    def gramschmidt(self):
        """
        Gram-Schmidt orthogonalization of an object of the class. 
        The method returns a matrice 'orthogonalmatrix' that is an 
        orthogonal basis of range(A).
        """
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
        
    def householder(self):
        """
        Will not work with a already triangular matrix (:
        """
        A = self.givenmatrix
        m = shape(A)[0]
        n = shape(A)[1]
        Q = identity(m)
        for i in range(n):
            Q_i = identity(m)
            x = A[i:m,i]
            s = int(sign(x[0]))
            u = s*array([norm(x)]+(m-i-1)*[0.]) 
            v_i = x + u
            v_i /= norm(v_i)
            Q_i_hat = eye(m-i) - 2*outer(v_i,v_i)
            Q_i[i:m,i:m] = Q_i_hat
            Q = Q_i@Q
            A = Q_i@A
        return Q, A
        
    def givens(self):
        """
        Takes a (mxn)-matrix with m≥n and returns its QR-factorization as the
        two matrices Q, A, by using Givens rotations.
        """
        A = self.givenmatrix
        m = shape(A)[0]
        n = shape(A)[1]
        Q = identity(m)
        for i in range(n): #counting through the columns of the matrix
            Q_i = identity(m)
            x = A[i:m,i]
            l = len(x)
            Q_i_hat = identity(l)
            for j in range(l-1):
                J_j = identity(l) #rotation matrix to be
                a = x[l-(j+2)]
                b = x[l-(j+1)]
                r = sqrt(a**2+b**2)
                c = a/r
                s = -b/r
                rotation = array([[c, -s], [s, c]])
                J_j[l-2-j:l-j, l-2-j:l-j] = rotation #rotation matrix in the 
                                                    #(n-(i-1), n-i)-plane
                
                Q_i_hat = J_j@Q_i_hat #matrix for all the rotations of one vector
            Q_i[i:m,i:m] = Q_i_hat
            Q = Q_i@Q
            A = Q_i@A
        print(Q@A)
        return Q, A
        
    def norm(self, matrix):
        """
        Returns the 2-norm of a the input matrix A.
        """
        return np.linalg.norm(matrix, ord=2)
        
        
    def qtq(self, matrix):
        """
        Returns the matrix product of the transpose of the input matrix A with
        itself.
        """
        return np.dot(matrix.transpose(),matrix)
        
    def deviation(self, matrix):
        """
        Returns the deviation of the output matrix of qtq from the identity 
        matrix.
        """
        qtq = self.qtq(matrix)
        I = np.identity(len(qtq))
        return self.norm(I-qtq)
        
    def allclose(self, matrix):
        """
        Returns True/False if all the entries of QTQ are closer than a 
        certain tolerance to the identity matrix.
        """
        qtq = self.qtq(matrix)
        I = np.identity(len(qtq))
        return np.allclose(qtq, I)
        
    def eigenvalues(self, matrix):
        """
        Returns an array with the eigenvalues of qtq of the input matrix A.
        """
        return la.eigvals(self.qtq(matrix))
    
    def determinant(self, matrix):
        """
        Returns the determinant of qtq of the input matrix A.
        """
        return la.det(self.qtq(matrix))

A0 = np.random.rand(8,4)
print('Random matrix', A0)
A = Orthogonalization(A0)

V = A.gramschmidt()
print('Norm of V', A.norm(V))
print('Deviation of VTV from I', A.deviation(V))
print('VTV allclose to I?', A.allclose(V))
print('Eigenvalues of VTV:', A.eigenvalues(V))
print('Determinant of VTV:', A.determinant(V))


Q,R = A.householder()
print('Triangular matrix:', R)
print('Q@R:', Q@R)
print('Norm of Q:', A.norm(Q))
print('Deviation of QTQ from I:', A.deviation(Q))
print('QTQ allclose to I?', A.allclose(Q))
print('Eigenvalues of QTQ:', A.eigenvalues(Q))
print('Determinant of QTQ:', A.determinant(Q))

B,S = la.qr(A0)

print('Triangular matrix:', S)
print('B@S',B@S)
print('Norm B:', A.norm(B))
print('Dev of BTB from I:', A.deviation(B))
print('BtB close to I?', A.allclose(B))
print('Eigenvalues of BTB:', A.eigenvalues(B))
print('Determinant of BTB:', A.determinant(B))

q,r = A.givens()
print('Triangular matrix:', r)
print('q@r:',q@r)
print('Norm:', A.norm(q))
print('Deviation from I of qtq:', A.deviation(q))
print('Qtq allclose to I?', A.allclose(q))
print('Eigenvalues of qtq:', A.eigenvalues(q))
print('Determinant of qtq:', A.determinant(q))