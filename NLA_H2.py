# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:58:41 2017

@author: edadagasan
"""
from scipy import *
from pylab import *


class Orthogonalization(object):
    '''
    A class of matrices with methods that orthogonalize the objects with
    different algorithms and that evaluate the result in different ways.
    '''
    
    def __init__(self, A):
        self.A = A
        
    def gramschmidt(self):
        '''
        Gram-Schmidt orthogonalization of an object of the class. 
        The method returns a matrice Q that is an orthogonal basis of
        range(A).
        '''
        Q=empty_like(self.A)
        for i in range(shape(Q)[1]):
            v_i=self.A[:,i]
            for j in range(i):
                v_i = v_i - inner(Q[:,j],self.A[:,i])*Q[:,j]
            #if norm(v_i) < 1.e-15 :
             #   Q[:,i] = v_i
            #else:
                Q[:,i] = v_i/norm(v_i)
       # for i in range(shape(Q)[1]-1):
        #    if norm(Q[:,i]) < 1.e-15 :
         #       Q=delete(Q, i, axis = 1)
        return Q
        
    def norm(self):
        """
        Returns the 2-norm of a the input matrix A.
        """
    
        return norm(self.gramschmidt(), ord=2)
        
    def qtq(self):
        """
        Returns the matrix product of the transpose of the input matrix A with
        itself.
        """
        A = self.A
        return transpose(A)@A
    
    def dev(self):
        """
        Returns the deviation of the output matrix of qtq from the identity 
        matrix.
        """
        QTQ = Orthogonalization(self.gramschmidt()).qtq()
        dev = norm(QTQ-identity(shape(QTQ)[0]), ord=2)
        return(dev)
        
    def eigenvalues(self):
        """
        Returns an array with the eigenvalues of qtq of the input matrix A.
        """
        Q = self.gramschmidt()
        return eigvals(Orthogonalization(Q).qtq())
        
    def det(self, matrix):
        """
        Returns the determinant of qtq of the input matrix A.
        """
        return(det(Orthogonalization(self.gramschmidt()).qtq()))
    
    def allclose(self):
        Q = self.gramschmidt()
        QTQ = Orthogonalization(Q).qtq()
        I = identity(shape(Q)[0])
        return allclose(A,I)
    
        
#M=array([[1,1,0,3], [0,0,0,5], [0,0,1,0]])
#Ma=Orthogonalization(M)
#Q=Ma.gramschmidt()
        
A = rand(50,50)
B = Orthogonalization(A)


control_gram = (B.norm(), B.dev(), B.allclose(), B.eigenvalues(), B.det())
print(control_gram)

q_py = Orthogonalization(qr(A)[0])

control_py = (q_py.norm(), q_py.dev(), q_py.allclose(), q_py.eigenvalues(), q_py.det(A))

print(control_py)
