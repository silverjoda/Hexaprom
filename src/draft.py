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
### Pytorch Basics, use Debugger to inspect tensors ###

# Terminology: Tensor = any dimensional matrix

# Empty tensor of shape 5x3. Notice the initial values are (sometimes) whatever garbage was in memory.
x = torch.empty(5, 3)
print(x)

# Construct tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

# Convert numpy arrays to pytorch
nparr = np.array([1,2])
x = torch.from_numpy(nparr)

# Convert pytorch arrays into numpy
nparr = x.numpy()

# Make operation using tensors (they support classic operators)
a = torch.tensor([3.])
b = torch.rand(1)
c = a + b # a + b = torch.add(a,b)
print("a: {}, b: {}, a + b: {}".format(a,b,c))

# Note, when performing operations make sure
# that the operands are of the same data type
a = torch.tensor(3)  # int64 type
b = torch.tensor(3.) # float32 type
print("atype: {}, btype: {}".format(a.dtype,b.dtype))

# ERROR, data type mismatch:
print(a + b)

# Convert data type
b = torch.tensor(3., dtype=torch.int32) # Explicitly force datatype
b = b.long()
print(a + b)

### Autograd ###


pass
