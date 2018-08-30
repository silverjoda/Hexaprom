import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc1(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def matchsignal(net, iters, sig):

    N = len(sig)

    # Torch objects
    lossfun = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    # Train iters times
    for i in range(iters):

        # Initial values
        s = torch.tensor(0., requires_grad=True)
        out = torch.tensor(0., requires_grad=True)

        outputs = []
        states = []

        # Single iteration of signal generation
        for j in range(N):
            x = torch.tensor([s, out], dtype=torch.float32)
            out, s = net(x)
            outputs.append(out)
            states.append(s)

        # Zero the gradient buffers and calculate loss
        optimizer.zero_grad()
        loss = lossfun(torch.stack(outputs), torch.from_numpy(sig).float())

        # Backprop
        loss.backward()

        # Apply gradients
        optimizer.step()

        if i % 50 == 0:
            print("Iter {}/{}, loss: {}".format(i, iters, loss))


def eval(net, sig):

    N = len(sig)
    pattern = np.fft.rfft(sig)
    real = np.abs(pattern.real)
    imag = pattern.imag

    # Initial values
    s = 0
    out = 0

    outputs = []
    states = []

    for j in range(N):
        x = torch.tensor([s, out], dtype=torch.float32)
        out, s = net(x)
        outputs.append(out.item())
        states.append(s.item())

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
    plt.plot(np.arange(N), outputs, 'r')


    plt.subplot(715)
    plt.ylabel('pred_real')
    plt.plot(np.arange(N/2 + 1), pred_real, 'k')

    plt.subplot(716)
    plt.ylabel('pred_imag')
    plt.plot(np.arange(N / 2 + 1), pred_imag, 'r')

    plt.subplot(717)
    plt.ylabel('states')
    plt.plot(np.arange(N), states, 'r')

    plt.show()


# Define parameters
N = 30
ITERS = 1000

# Make oscillator model
osc = Net()
print(osc)

# Make signal
F1 = 0.9
F2 = 0.4
sig = np.sin(F1 * np.arange(N))

# Train
try:
    matchsignal(osc, ITERS, sig)
except KeyboardInterrupt:
    print("User interrupted process.")

#Evaluate result
eval(osc, sig)
