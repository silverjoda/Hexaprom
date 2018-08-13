import numpy as np
import gym
import cma
np.random.seed(0)

# the function we want to optimize
def f(w):

    reward = 0
    step = 0
    done = False
    env_obs = env.reset()
    prev_torques = [0,0]

    while not done:

        # Make actions:

        # Oscillator 0
        o0 = list(env_obs) + prev_torques
        a0, m0 = np.tanh(np.matmul(o0, w[0:n1].reshape((10,2))))

        # Oscillator 1
        o1 = list(env_obs) + prev_torques
        a1, m1 = np.tanh(np.matmul(o1, w[n1:n1 + n2].reshape((10,2))))

        mult = 1
        offset = 0

        #m0 = m1 = a0 = a1 = 1
        #print(a0, m0, a1, m1)

        t0 = (offset + mult * a0) * np.sin(m0 * step)
        t1 = (offset + mult * a1) * np.sin(m1 * step)

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t1])

        if animate:
            env.render()

        reward += rew
        step += 0.03

        prev_torques = [t0, t1]

    return -reward

# Make environment
env = gym.make("Swimmer-v2")
print("Action space: {}, observation space: {}".format(env.action_space.shape, env.observation_space.shape))
animate = True

# Observations
# x, y, th, j1, j2, dx, dy, dth, dj1, dj2

# Generate weights
n1 = (10 * 2)
n2 = (10 * 2)

N_weights = n1 + n2
w = np.random.randn(N_weights)

es = cma.CMAEvolutionStrategy(w, 0.5)

try:
    es.optimize(f, iterations=2000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()

print(es.result.xbest, file=open("swimmer_weights.txt", "a"))

