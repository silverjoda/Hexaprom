import numpy as np
import gym
import cma
from weight_distributor import Wdist
#np.random.seed(0)
from time import sleep
import matplotlib.pyplot as plt

def f(w):

    out, state = np.random.randn() * 0.1

    outputs = []

    for _ in range(100):

        # fl
        l = np.tanh(np.matmul([out, state], wdist.get_w('w1', w)) + wdist.get_w('b1', w))
        out, state = np.matmul(l, wdist.get_w('w2', w)) + wdist.get_w('b2', w)

        # Step environment
        outputs.append(out)

    pred_pattern = np.fft.fft(np.array(outputs))
    loss = np.mean(np.square(pred_pattern - pattern))

    return loss


# Generate weights
wdist = Wdist()
wdist.addW((2, 3), 'w1')
wdist.addW((3,), 'b1')
wdist.addW((3, 2), 'w2')
wdist.addW((2,), 'b2')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
W_MULT = 1
ACT_MULT = 1

N = 70
F = 0.15

# Make frequency pattern
sig = np.sin(F * np.arange(N))
pattern = np.fft.fft(sig)

plt.figure(1)
plt.subplot(311)
plt.ylabel('sig')
plt.plot(np.arange(N), sig, 'k')

plt.subplot(312)
plt.ylabel('real')
plt.plot(np.arange(N/2), pattern.real[0:N//2], 'b')

plt.subplot(313)
plt.ylabel('imag')
plt.plot(np.arange(N/2), pattern.imag[0:N//2], 'r')

plt.show()

w = np.random.randn(N_weights) * W_MULT
es = cma.CMAEvolutionStrategy(w, 0.7)
try:
    es.optimize(f, iterations=1000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("sin_weights.txt", "a"))
