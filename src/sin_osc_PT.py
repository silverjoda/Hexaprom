import numpy as np
import torch
import torch.nn as nn
import pytorch_fft.fft.autograd as fft
import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
        #return self.fc1(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def matchsignal(net, iters, sig):

    N = len(sig)
    tsig = torch.from_numpy(sig)

    # Torch objects
    lossfun = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)

    # Train iters times
    for i in range(iters):

        # Initial values
        x = torch.tensor([0, 0, 0], dtype=torch.float32, requires_grad=True)
        xlist = []

        # Single iteration of signal generation
        for j in range(N):
            x = net(x)
            xlist.append(x)

        xlist = torch.stack(xlist)

        optimizer.zero_grad()
        loss = lossfun(xlist[:, 0], tsig)
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % 50 == 0:
            print("Iter {}/{}, loss: {}".format(i, iters, loss))


def eval(net, sig):

    N = len(sig)
    pattern = np.fft.rfft(sig)
    real = np.abs(pattern.real)
    imag = pattern.imag

    # Initial values
    s1 = 0
    s2 = 0
    out = 0

    outputs = []
    states = []

    for j in range(N):
        x = torch.tensor([out, s1, s2], dtype=torch.float32)
        out, s1, s2 = net(x)
        outputs.append(out.item())
        states.append(s1.item())

    pattern_pred = np.fft.rfft(np.array(outputs))
    pred_real = np.abs(pattern_pred.real)
    pred_imag = pattern_pred.imag

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
    plt.ylabel('states')
    plt.plot(np.arange(N), states, 'r')

    plt.show()


# Define parameters
N = 30
ITERS = 2000

# Make oscillator model
osc = Net()
print(osc)

# Make signal
F1 = 0.95
F2 = 0.3
sig = np.sin(F1 * np.arange(N, dtype=np.float32)) + np.sin(F2 * np.arange(N, dtype=np.float32))

# Train
try:
    matchsignal(osc, ITERS, sig)
except KeyboardInterrupt:
    print("User interrupted process.")

#Evaluate result
eval(osc, sig)
