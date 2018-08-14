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

        # Oscillator 0
        o0 = list(env_obs) + prev_torques
        t0 = np.tanh(np.matmul(o0, w[0:n1_a].reshape((14, 3))))
        t0 = mult * np.matmul(t0, w[n1_a:n1_a + n1_b].reshape((3, 1)))

        # Oscillator 1
        o1 = list(env_obs) + prev_torques
        t1 = mult * np.matmul(o1, w[n1:n1 + n2].reshape((14,3)))

        # Oscillator 2
        o2 = list(env_obs) + prev_torques
        t2 = mult * np.matmul(o2, w[n1 + n2:].reshape((14,3)))

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t1, t2])

        if animate:
            env.render()

        reward += rew
        prev_torques = [t0, t1, t2]

    return -reward

# Make environment
env = gym.make("Hopper-v2")
animate = False

# Generate weights
n1_a = (14 * 3)
n1_b = (3 * 1)
n1 = n1_a + n1_b

n2_a = (14 * 3)
n2_b = (3 * 1)
n2 = n2_a + n2_b

n3_a = (14 * 3)
n3_b = (3 * 1)
n3 = n3_a + n3_b

N_weights = n1 + n2 + n3
W_MULT = 0.3
mult = 3
w = np.random.randn(N_weights) * W_MULT

print("N_weights: {}, mult {}, n_mult {}".format(N_weights, mult, W_MULT))

es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=10000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("hopper_weights.txt", "a"))
