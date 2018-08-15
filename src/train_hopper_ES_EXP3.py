import numpy as np
import gym
import cma
np.random.seed(0)

# EXP: double layer NN, sparser observations (less parameters), heirarchical control

# the function we want to optimize
def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    prev_torques = [0,0,0]

    while not done:

        # Observations:
        # 0,  1,  2,  3,  4,  5,  6,   7,   8,   9,  10
        # z, th, j1, j2, j3, dx, dz, dth, dj1, dj2, dj3

        # Oscillator 0
        o0 = list(env_obs[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]) + prev_torques[0:2]
        t0 = np.tanh(np.matmul(o0, w[0:n1_a].reshape((13, 3))))
        t0, s0 = mult * np.tanh(np.matmul(t0, w[n1_a:n1].reshape((3, 2))))

        # Oscillator 1
        o1 = list(env_obs[[2,3,4]]) + [t0, s0] + prev_torques
        t1 = np.tanh(np.matmul(o1, w[n1:n1+n2_a].reshape((8, 3))))
        t1, s1 = mult * np.tanh(np.matmul(t1, w[n1+n2_a:n1 + n2].reshape((3, 2))))

        # Oscillator 2
        o2 = list(env_obs[[3,4]]) + [t1, s1] + prev_torques[1:]
        t2 = np.tanh(np.matmul(o2, w[n1 + n2:n1 + n2 + n3_a].reshape((6, 2))))
        t2 = mult * np.tanh(np.matmul(t2, w[n1 + n2 + n3_a:n1 + n2 + n3].reshape((2, 1))))

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
n1_a = (13 * 3)
n1_b = (3 * 2)
n1 = n1_a + n1_b

n2_a = (8 * 3)
n2_b = (3 * 2)
n2 = n2_a + n2_b

n3_a = (6 * 2)
n3_b = (2 * 1)
n3 = n3_a + n3_b

N_weights = n1 + n2 + n3
W_MULT = 0.3
ES_STD = 0.5
mult = 3
w = np.random.randn(N_weights) * W_MULT

print("N_weights: {}, mult {}, n_mult {}, ES_ESD {}".format(N_weights, mult, W_MULT, ES_STD))

es = cma.CMAEvolutionStrategy(w, ES_STD)
try:
    es.optimize(f, iterations=20000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("hopper_weights.txt", "a"))
