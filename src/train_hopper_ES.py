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
        o0 = list(env_obs) + prev_torques
        a0, m0 = np.tanh(np.matmul(o0, w[0:n1].reshape((14,2))))

        # Oscillator 1
        o1 = list(env_obs) + prev_torques
        a1, m1 = np.tanh(np.matmul(o1, w[n1:n1 + n2].reshape((14,2))))

        # Oscillator 2
        o2 = list(env_obs) + prev_torques
        a2, m2 = np.tanh(np.matmul(o2, w[n1 + n2:].reshape((14,2))))

        mult = 1

        r0 = (mult * a0) * np.sin(m0 * step)
        r1 = (mult * a1) * np.sin(m1 * step)
        r2 = (mult * a2) * np.sin(m2 * step)

        j0,j1,j2 = env_obs[[2,3,4]]

        kp = 3
        t0 = kp * (r0 - j0)
        t1 = kp * (r1 - j1)
        t2 = kp * (r2 - j2)

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t1, t2])

        # TODO: Inspect oscillator, action values etc
        # TODO: Try optimize normal mlp policy but predicting oscillator values instead of torques

        if animate:
            env.render()

        reward += rew
        step += 0.01

        prev_torques = [r0, r1, r2]

    return -reward

# Make environment
env = gym.make("Hopper-v2")
animate = True

# Generate weights
n1 = (14 * 2)
n2 = (14 * 2)
n3 = (14 * 2)

N_weights = n1 + n2 + n3
W_MULT = 0.05
w = np.random.randn(N_weights) * W_MULT

es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=2000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("hopper_weights.txt", "a"))
