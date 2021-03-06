import numpy as np
import gym
import cma
from weight_distributor import Wdist
#np.random.seed(0)
from time import sleep
import matplotlib.pyplot as plt

def f(w, plot=False):

    # TODO: Double oscillator

    # Make frequency pattern
    N = 140
    F1 = 0.9
    F2 = 0.4
    sig = np.sin(F1 * np.arange(N)) + np.sin(F2 * np.arange(N))
    pattern = np.fft.rfft(sig)
    real = pattern.real
    imag = pattern.imag

    out1, out2, state1, state2 = [0,0,0,0]

    outputs = np.zeros((3, N))
    states = np.zeros((2, N))

    for i in range(N):

        # fl
        o1, s1  = np.matmul(np.array([out1, out2, state1]), wdist.get_w('w1', w)) + wdist.get_w('b1', w)
        o2, s2 = np.matmul(np.array([out2, out1, state2]), wdist.get_w('w2', w)) + wdist.get_w('b2', w)

        out1 = o1; out2 = o2; state1 = s1; state2 = s2; out = out1 + out2

        # Step environment
        outputs[:, i] = [out1, out2, out]
        states[:, i] = [s1, s2]

    pred_pattern = np.fft.rfft(outputs[2,:])
    pred_real = pred_pattern.real
    pred_imag = pred_pattern.imag

    loss = np.mean(np.square(pred_real - real)) + np.mean(np.square(pred_imag - imag))
    #loss = np.mean(np.square(np.array(outputs[2,:]) - sig))

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
        plt.plot(np.arange(N), outputs[0,:], 'r')
        plt.plot(np.arange(N), outputs[1,:], 'g')
        plt.plot(np.arange(N), outputs[2,:], 'k')

        plt.subplot(715)
        plt.ylabel('pred_real')
        plt.plot(np.arange(N/2 + 1), pred_real, 'r')

        plt.subplot(716)
        plt.ylabel('pred_imag')
        plt.plot(np.arange(N / 2 + 1), pred_imag, 'r')

        plt.subplot(717)
        plt.ylabel('states')
        plt.plot(np.arange(N), states[0, :], 'r')
        plt.plot(np.arange(N), states[1, :], 'g')

        plt.show()

    return loss


# Generate weights
wdist = Wdist()
wdist.addW((3, 2), 'w1')
wdist.addW((2,), 'b1')

wdist.addW((3, 2), 'w2')
wdist.addW((2,), 'b2')


N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
W_MULT = 1
ACT_MULT = 1

w = np.random.randn(N_weights) * W_MULT
es = cma.CMAEvolutionStrategy(w, 0.8)

try:
    es.optimize(f, iterations=3000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()

for i in range(1):
   f(es.result.xbest, plot=True)

print('[' + ','.join(map(str, es.result.xbest)) + ']', file=open("sin_weights.txt", "a"))
