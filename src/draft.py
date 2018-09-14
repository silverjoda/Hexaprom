import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    return np.maximum(x,0)

z = [2,1,0,-1,-0.2]
print(z)
print(relu(z))