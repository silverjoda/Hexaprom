import numpy as np
import gym
import cma
np.random.seed(0)

# the function we want to optimize
def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    prev_torques = [0,0,0]

    while not done:

        # Make actions:

        mult = 2

        # Oscillator 0
        o0 = list(env_obs) + prev_torques
        t0 = mult * np.tanh(np.matmul(o0, w[0:n1].reshape((14,1))))

        # Oscillator 1
        o1 = list(env_obs) + prev_torques
        t1 = mult * np.tanh(np.matmul(o1, w[n1:n1 + n2].reshape((14,1))))

        # Oscillator 2
        o2 = list(env_obs) + prev_torques
        t2 = mult * np.tanh(np.matmul(o2, w[n1 + n2:].reshape((14,1))))

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t1, t2])

        if animate:
            env.render()

        reward += rew
        prev_torques = [t0, t1, t2]

    return -reward

# Make environment
env = gym.make("Ant-v3")
animate = True

# Generate weights
n1 = (14 * 1)
n2 = (14 * 1)
n3 = (14 * 1)

N_weights = n1 + n2 + n3
W_MULT = 1
w = np.random.randn(N_weights) * W_MULT

es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=10000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("hopper_weights.txt", "a"))
