#!/usr/bin/env python
# coding: utf-8

# Tip: To execute a specific block of code from pycharm:
# a) open a python console (Tools -> python console),
# b) highlight code
# c) alt + shift + e to execute

import numpy as np

### Simple numerical manipulation recap. ###

# Numbers
a = 3; b = 2; c = 1
print("a: {}, b: {}, c: {}, a * b + c: {}".format(a,b,c, a * b + c))


# Vectors
a = np.array([10,40,20]); b = np.array([1,0.1,0.01]); c = 1.
print("a: {}, b: {}, c: {}, a * b + c: {}".format(a,b,c, a * b + c)) # Elementwise multiplication

a = np.array([[10,40,20]]); b = np.array([1,0.1,0.01]); c = 1.
print("a.shape: {}, b.shape: {}".format(a.shape, b.shape)) # Shapes of vectors as numpy sees it
print("a.T * b + c:") # Vector multiplication.
print(a.T * b + c)

# Matrices
A = np.eye(3); B = np.random.randn(3,3) # Identity and random 3x3 matrices

print("A: ")
print(A)

print("B: ")
print(B)

print("A * B: ")
print(A * B) # Elementwise multiplication

print("AB: ")
print(A @ B) # Matrix multiplication,  A @ B == A.dot(B) == np.matmul(A,B)

# Note: Never use for loops to implement any sort of vector or matrix multiplication!

import torch
### Pytorch Basics, use Debugger for clarity ###

# Empty matrix of shape 5x3. Notice the initial values are (sometimes) whatever garbage was in memory.
x = torch.empty(5, 3)
print(x)


# Random matrix of shape 5x3 
x = torch.rand(5, 3)
print(x)

