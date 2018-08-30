import numpy as np
import gym
import cma
from weight_distributor import Wdist
#np.random.seed(0)
from time import sleep
import matplotlib.pyplot as plt

def matchsig(siga, sigb):
    assert len(siga) == len(sigb)
    n = len(siga)
    best = 1e10
    for i in range(n):
        loss = np.mean(np.square(siga - np.roll(sigb, i)))
        if loss < best: best = loss
    return best

def f(w, plot=False):

    # Make frequency pattern
    N = 30
    F = 2
    sig = np.sin(F * np.arange(N))
    pattern = np.fft.rfft(sig)
    real = np.abs(pattern.real)
    imag = pattern.imag

    out, state = [0,0]

    outputs = []
    states = []

    for _ in range(N):

        # fl
        out, state = np.matmul(np.array([out, state]), wdist.get_w('w1', w)) + wdist.get_w('b1', w)

        # Step environment
        outputs.append(out)
        states.append(state)

    pred_pattern = np.fft.rfft(np.array(outputs))
    pred_real = np.abs(pred_pattern.real)
    pred_imag = pred_pattern.imag

    loss = np.mean(np.square(pred_real - real)) + np.mean(np.square(pred_imag - imag))
    #loss = np.mean(np.square(np.array(outputs) - sig))

    if plot:
        plt.figure(1)
        plt.subplot(711)
        plt.ylabel('sig')
        plt.plot(np.arange(N), sig, 'k')

        plt.subplot(712)
        plt.ylabel('sig_real')
        plt.plot(np.arange(N / 2 + 1), real, 'b')

        plt.subplot(713)
        plt.ylabel('sig_imag')
        plt.plot(np.arange(N / 2 + 1), imag, 'b')

        plt.subplot(714)
        plt.ylabel('predicted_sig')
        plt.plot(np.arange(N), outputs, 'k')

        plt.subplot(715)
        plt.ylabel('pred_real')
        plt.plot(np.arange(N/2 + 1), pred_real, 'r')

        plt.subplot(716)
        plt.ylabel('pred_imag')
        plt.plot(np.arange(N / 2 + 1), pred_imag, 'r')

        plt.subplot(717)
        plt.ylabel('state')
        plt.plot(np.arange(N), states, 'g')

        plt.show()

    return loss


# Generate weights
wdist = Wdist()
wdist.addW((2, 2), 'w1')
wdist.addW((2,), 'b1')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
W_MULT = 1
ACT_MULT = 1

w = np.random.randn(N_weights) * W_MULT
es = cma.CMAEvolutionStrategy(w, 0.7)

try:
    es.optimize(f, iterations=7000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()

for i in range(1):
   f(es.result.xbest, plot=True)

print('[' + ','.join(map(str, es.result.xbest)) + ']', file=open("sin_weights.txt", "a"))
