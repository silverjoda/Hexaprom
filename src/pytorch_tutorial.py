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
#print(a + b)

# Convert data type
b = torch.tensor(3., dtype=torch.int32) # Explicitly force datatype
b = b.long()
print(a + b)

### Autograd ###

# Make two tensors
a = torch.tensor(8., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

# c tensor is a function of a,b
c = torch.exp((a / 2.) - b * b + 0.5)
print("c value before applying gradients: {}".format(c))

# Backwards pass
c.backward()

# Print gradients of individual tensors which contributed to c
print("a.grad: {}, b.grad: {}".format(a.grad, b.grad))

# Move a,b towards the direction of greatest increase of c.
a = a + a.grad
b = b + b.grad
c = torch.exp((a / 2.) - b * b + 0.5)
print("c value after applying gradients: {}".format(c))

# If we don't want to track the history of operations then
# the torch.no_grad() context is used to create tensors and operations
print(a.requires_grad)
with torch.no_grad():
    print((a + 2).requires_grad)

# Note: Whenever we create a tensor or perform an operation requiring
# a gradient, a node is added to the operation graph in the background.
# This graph enables the backwards() pass to be called on any node and find all
# ancestor operations.


### Simple regressor: Optimizing bivariate rosenbrock function ###
### Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2 with global
### Min at (x,y) = (a,a^2). For a=1, b=100 this is (1,1)

# Clear all previous variables
import sys
sys.modules[__name__].__dict__.clear()

# Import everything
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define variables that we are minimizing
a = torch.tensor(1., requires_grad=False)
b = torch.tensor(100., requires_grad=False)
x = torch.randn(1, requires_grad=True)
y = torch.randn(1, requires_grad=True)

# Learning rate and iterations
lr = 0.001
iters = 10000

def rb_fun(x,y,a,b):
    return (a - x) ** 2 + b * (y - x ** 2) ** 2

print("Initial values: x: {}, y: {}, f(x,y): {}".format(x,y,rb_fun(x,y,a,b)))

for i in range(iters):
    # Forward pass:
    loss = rb_fun(x,y,a,b)

    # Backward pass
    loss.backward()

    # Apply gradients
    with torch.no_grad():
        x -= x.grad * lr
        y -= y.grad * lr

        # Manually zero the gradients after updating weights
        x.grad.zero_()
        y.grad.zero_()

    if i > 0 and i % 10 == 0:
        print("Iteration: {}/{}, x,y: {},{}, loss: {}".format(i + 1, iters, x[0], y[0], loss[0]))


### Convolutional Neural network 3D pose estimation ###

# Load dataset.  M: amount of examples, h: height, w: width, 3: rgb channels, d: label dimension
Xtrn = np.load("Xtrn.npy") # shape: (M,h,w,3)
Ytrn = np.load("Ytrn.npy") # shape: (M,d)
Xtst = np.load("Xtst.npy") # shape: (M,h,w,3)
Ytst = np.load("Ytst.npy") # shape: (M,d)


# Get shape of images
M, h, w, c = Xtrn.shape
_, d = Ytrn.shape

# Check that we have 3 channels
assert c == 3

#Manually define batchsize and number of training iterations and lr
N = 32; iters = 1000; lr = 1e-3

# Make input variables
x = torch.empty(N, h, w, c)
y = torch.empty(N, d)

cnn = nn.Sequential(nn.Conv2d(3, 16, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=1),
                    nn.ReLU(),
                    nn.Dropout(p=0.3),
                    nn.Conv2d(64, d, kernel_size=1, stride=1))

# Loss function
loss_fn = torch.nn.MSELoss()

# Optimizer: Adam
optim = torch.optim.Adam(cnn.parameters(), lr=lr)

# Train
for i in range(iters):

    # Get data batch of N samples
    rnd_vec = np.random.choice(np.arange(M), N, replace=False)
    X, Y = Xtrn[rnd_vec], Ytrn[rnd_vec]

    # Make prediction
    Y_pred = cnn.forward(X)

    # Compute loss
    loss = loss_fn(Y_pred, Y)

    # Backprop
    loss.backward()

    # Update step
    optim.step()

    if i % 10 == 0:
        print("Iters: {}/{}, loss: {}".format(i, iters, loss))

# Test
Y_pred = cnn.forward(Xtst)
testloss = loss_fn(Y_pred, Ytst)
print("Test loss: {}".format(testloss))

# Print predictions for random examples
for i in range(20):
    rnd = np.random.randint(0, Y_pred.shape[0])
    print("cnn predicted: {}, ground truth result: {}, error_vec: {}".format(Y_pred[rnd], Ytst[rnd], Y_pred[rnd] - Ytst[rnd]))
