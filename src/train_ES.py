import numpy as np
np.random.seed(0)

def nn(w):
    def forward_pass(x):
        return np.matmul(w[0:5])
    return forward_pass

def genweights(nn_struct):
    pass

# the function we want to optimize
def f(w):

    # ... 1) create a neural network with weights w
    policy = nn(w)

    # ... 2) run the neural network on the environment for some time
    # ... 3) sum up and return the total reward

    reward = 0
    return reward

# hyperparameters
npop = 50 # population size
sigma = 0.1 # noise standard deviation
alpha = 0.001 # learning rate

# Define nn structure
input_dim = 8
act_dim = 8
nn_struct = [input_dim, act_dim, 5, 5]

# Generate weights from nn_struct


w = np.random.randn(3) # our initial guess is random
for i in range(300):

    # print current fitness of the most likely parameter setting
    if i % 20 == 0:
    print('iter %d. w: %s, solution: %s, reward: %f' %
          (i, str(w), str(solution), f(w)))

    # initialize memory for a population of w's, and their rewards
    N = np.random.randn(npop, 3) # samples from a normal distribution N(0,1)
    R = np.zeros(npop)
    for j in range(npop):
        w_try = w + sigma*N[j] # jitter w using gaussian of sigma 0.1
        R[j] = f(w_try) # evaluate the jittered version

        # standardize the rewards to have a gaussian distribution
        A = (R - np.mean(R)) / np.std(R)
        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        w = w + alpha/(npop*sigma) * np.dot(N.T, A)