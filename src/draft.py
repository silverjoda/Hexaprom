import torch as T

a = T.rand(1,32)
b = a.view((1, 4, -1))
print(a)
print(a.shape)
print(b)
print(b.shape)


