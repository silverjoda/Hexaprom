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
    prev_torques = [0,0,0]

    while not done:

        # Make actions:

        # Oscillator 0
        o0 = list(env_obs[0:4]) + prev_torques[0:1]
        a0, m0 = np.maximum(np.matmul(o0, w[0:n1].reshape((5,2))), 0)

        # Oscillator 1
        o1 = list(env_obs[2:5]) + prev_torques[1:2]
        a1, m1 = np.maximum(np.matmul(o1, w[n1:n1 + n2].reshape((4,2))), 0)

        # Oscillator 2
        o2 = list(env_obs[3:5]) + prev_torques[2:3]
        a2, m2 = np.maximum(np.matmul(o2, w[n1 + n2:].reshape((3,2))), 0)

        t0 = (0.5 + 5 * a0) * np.sin(m0 * step)
        t1 = (0.5 + 5 * a1) * np.sin(m1 * step)
        t2 = (0.5 + 5 * a2) * np.sin(m2 * step)

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t1, t2])

        # TODO: Inspect oscillator, action values etc
        # TODO: Try optimize normal mlp policy but predicting oscillator values instead of torques

        if animate:
            env.render()

        reward += rew
        step += 0.005

        prev_torques = [t0, t1, t2]

    return -reward

# Make environment
env = gym.make("Hopper-v2")
animate = True

# Generate weights
n1 = (5 * 2)
n2 = (4 * 2)
n3 = (3 * 2)

N_weights = n1 + n2 + n3
w = np.random.randn(N_weights)

es = cma.CMAEvolutionStrategy(w, 0.5)
es.optimize(f, iterations=1000)
es.result_pretty()

animate = True
for i in range(10):
    f(es.result.xbest)

print("Best value: {}".format(es.result.fbest))


