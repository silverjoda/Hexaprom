import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

def f(w, plot=False):

    # Make frequency pattern
    N = 40
    F1 = 0.9
    F2 = 0.4
    sig = np.sin(F1 * np.arange(N)) + np.sin(F2 * np.arange(N))
    pattern = np.fft.rfft(sig)
    real = np.abs(pattern.real)
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
    pred_real = np.abs(pred_pattern.real)
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

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def matchsignal(net, iters, N):

    # Make signal
    F1 = 0.9
    F2 = 0.4
    sig = np.sin(F1 * np.arange(N))
    pattern = np.fft.rfft(sig)
    real = np.abs(pattern.real)
    imag = pattern.imag

    # Torch objects
    lossfun = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

    # Initial values
    s = 0
    out = 0

    outputs = []
    states = []

    # Train iters times
    for i in range(iters):

        # Single iteration of signal generation
        for j in range(N):
            out, s = net(np.expand_dims([s,out], axis=0))
            outputs.append(out)
            states.append(s)

        # Zero the gradient buffers and calculate loss
        optimizer.zero_grad()
        loss = lossfun(outputs, sig)

        # Backprop
        loss.backwards()

        # Apply gradients
        optimizer.step()

        if iters % 10 == 0:
            print("Iter {}/{}, loss: {}".format(i, iters, loss))


def eval(net, N):
    pass

# Define parameters
N = 50
ITERS = 1000

# Make oscillator model
osc = Net()

# Train
try:
    matchsignal(osc, ITERS, N)
except KeyboardInterrupt:
    print("User interrupted process.")

#Evaluate result
eval(osc, N)
