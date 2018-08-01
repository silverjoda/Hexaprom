import numpy as np
import gym
np.random.seed(0)

def make_nn(w):
    def forward_pass(x):
        return np.matmul(w[0:5])
    return forward_pass


# the function we want to optimize
def f(w):

    reward = 0
    step = 0
    done = False
    env_obs = env.reset()
    prev_torques = [0,0,0]

    while not done:

        # Make actions:

        # Oscillator 0
        o0 = env_obs[0:4] + [prev_torques[0]]
        a0, m0 = np.matmul(o0, w[0:n1])

        # Oscillator 1
        o1 = env_obs[2:5] + [prev_torques[1]]
        a1, m1 = np.matmul(o1, w[n1:n1 + n2])

        # Oscillator 2
        o2 = env_obs[3:5] + [prev_torques[2]]
        a2, m2 = np.matmul(o2, w[n1 + n2:])

        t0 = a0 * np.sin(m0 * step)
        t1 = a1 * np.sin(m1 * step)
        t2 = a2 * np.sin(m2 * step)

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t1, t2])

        if animate:
            env.render()

        reward += rew
        step += 0.01

        prev_torques = [t0, t1, t2]

    return reward

# hyperparameters
npop = 50 # population size
sigma = 0.1 # noise standard deviation
alpha = 0.001 # learning rate
N_iters = 1000
animate = True

# Make environment
env = gym.make("Hopper-v2")

# Generate weights
n1 = (5 * 2)
n2 = (4 * 2)
n3 = (3 * 2)

N_weights = n1 + n2 + n3
w = np.random.randn(N_weights)

for i in range(N_iters):

    # initialize memory for a population of w's, and their rewards
    N = np.random.randn(npop, N_weights) # samples from a normal distribution N(0,1)
    R = np.zeros(npop)
    for j in range(npop):
        w_try = w + sigma * N[j] # jitter w using gaussian of sigma 0.1
        R[j] = f(w_try) # evaluate the jittered version

        # standardize the rewards to have a gaussian distribution
        A = (R - np.mean(R)) / np.std(R)
        # perform the parameter update. The matrix multiply below
        # is just an efficient way to sum up all the rows of the noise matrix N,
        # where each row N[j] is weighted by A[j]
        w = w + alpha / (npop * sigma) * np.dot(N.T, A)

    # print current fitness of the most likely parameter setting
    if i % 10 == 0:
        print('iter %d. Reward: %f' %
              (i, f(w)))
