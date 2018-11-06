import numpy as np
import gym
import cma
from weight_distributor import Wdist
from time import sleep
import quaternion
import roboschool



def f_wrapper(env, wdist, animate):
    def f(w):
        reward = 0
        done = False
        env_obs = env.reset()

        while not done:

            # Observations
            l1 = np.tanh(np.matmul(np.asarray(env_obs), wdist.get_w('w_l1', w)) + wdist.get_w('b_l1', w))
            #l2 = np.matmul(l1, wdist.get_w('w_l2', w)) + wdist.get_w('b_l2', w)
            #l3 = np.matmul(l2, wdist.get_w('w_l3', w)) + wdist.get_w('b_l3', w)

            # Step environment
            env_obs, rew, done, _ = env.step(l1)

            if animate:
                env.render()

            reward += rew

        return -reward
    return f

def train(params):

    env_name, iters, n_hidden, animate = params

    env = gym.make(env_name)
    print("Env: {} Action space: {}, observation space: {}".format(env_name, env.action_space.shape,
                                                                   env.observation_space.shape))

    # Generate weight object
    wdist = Wdist()

    wdist.addW((env.observation_space.shape[0], 7), 'w_l1')
    wdist.addW((7,), 'b_l1')

    #wdist.addW((n_hidden, env.action_space.shape[0]), 'w_l2')
    #wdist.addW((env.action_space.shape[0],), 'b_l2')

    N_weights = wdist.get_N()
    print("Nweights: {}".format(N_weights))
    w = np.random.randn(N_weights)

    es = cma.CMAEvolutionStrategy(w, 0.5)

    print("Comments: n_hidden = {}".format(n_hidden))

    try:
        es.optimize(f_wrapper(env, wdist, animate), iterations=iters)
    except KeyboardInterrupt:
        print("User interrupted process.")

    return es.result.fbest

env_name = "SwimmerLong-v0"
train((env_name, 1000, 5, True))
exit()

params = []
reps = 3
hidden_max = 6
for h in range(hidden_max):
    for _ in range(reps):
        params.append((env_name, 400, h + 1, False))

from multiprocessing.dummy import Pool as ThreadPool
pool = ThreadPool(1)
results = pool.map(train, params)

print(results)

n_hid = []
for h in range(hidden_max):
    n_hid.append(np.mean(results[h*reps:h*reps + reps]))

print("Results for hidden layers 1-{}: {}".format(hidden_max, n_hid))

#print(es.result.xbest, es.result.fbest, sep=',', file=open("hopper_weights.txt", "a"))


