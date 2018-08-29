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
    N = 50
    F = 0.3
    sig = np.sin(F * np.arange(N))
    pattern = np.fft.rfft(sig)
    real = np.abs(pattern.real)

    out, state = [0,0] #np.random.randn(2) * 0.1

    outputs = []

    for _ in range(N):

        # fl
        l = np.tanh(np.matmul([out, state], wdist.get_w('w1', w)) + wdist.get_w('b1', w))
        out, state = np.matmul(l, wdist.get_w('w2', w)) + wdist.get_w('b2', w)

        # Step environment
        outputs.append(out)

    pred_pattern = np.fft.rfft(np.array(outputs))
    pred_real = np.abs(pred_pattern.real)

    loss = np.mean(np.square(pred_real[:5] - real[:5]))
    #loss = matchsig(sig, outputs)
    #loss = np.mean(np.square(np.array(outputs) - sig))


    if plot:
        plt.figure(1)
        plt.subplot(411)
        plt.ylabel('sig')
        plt.plot(np.arange(N), sig, 'k')

        plt.subplot(412)
        plt.ylabel('sig_real')
        plt.plot(np.arange(N / 2 + 1), real, 'b')

        plt.subplot(413)
        plt.ylabel('predicted_sig')
        plt.plot(np.arange(N), outputs, 'k')

        plt.subplot(414)
        plt.ylabel('pred_real')
        plt.plot(np.arange(N/2 + 1), pred_real, 'b')

        plt.show()

    return loss


# Generate weights
wdist = Wdist()
wdist.addW((2, 3), 'w1')
wdist.addW((3,), 'b1')
wdist.addW((3, 2), 'w2')
wdist.addW((2,), 'b2')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
W_MULT = 0.5
ACT_MULT = 1

w = np.random.randn(N_weights) * W_MULT
es = cma.CMAEvolutionStrategy(w, 0.5)

try:
    es.optimize(f, iterations=10000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()

for i in range(5):
    f(w, plot=True)

print(es.result.xbest, file=open("sin_weights.txt", "a"))
