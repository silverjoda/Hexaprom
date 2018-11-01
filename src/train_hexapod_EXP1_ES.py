import numpy as np
import gym
import cma
from weight_distributor import Wdist

def getStats(x_batch, w):

    l1_f = []
    l2_f = []
    for x in x_batch:
        l1,l2,_ = forward(x, w)
        l1_f.append(l1)
        l2_f.append(l2)

    l1_mu = np.mean(l1_f, axis=0)
    l1_std = np.mean([np.square(l - l1_mu) for l in l1_f], axis=0)

    l2_mu = np.mean(l2_f, axis=0)
    l2_std = np.mean([np.square(l - l2_mu) for l in l2_f], axis=0)

    return l1_mu, l1_std, l2_mu, l2_std


def forward(x, w, batchstats=None):
    # Observations
    l1 = np.tanh(np.matmul(np.asarray(x), wdist.get_w('w_l1', w)) + wdist.get_w('b_l1', w))

    if batchstats is not None:
        l1 = ( (l1 - batchstats['l1_mu']) / np.sqrt(batchstats['l1_std']) ) * batchstats['l1_gamma'] + batchstats['l1_beta']

    l2 = np.tanh(np.matmul(l1, wdist.get_w('w_l2', w)) + wdist.get_w('b_l2', w))

    if batchstats is not None:
        l2 = ( (l2 - batchstats['l2_mu']) / np.sqrt(batchstats['l2_std']) ) * batchstats['l2_gamma'] + batchstats['l2_beta']

    l3 = np.matmul(l2, wdist.get_w('w_l3', w)) + wdist.get_w('b_l3', w)
    return l1,l2,l3

def f(w):

    reward = 0
    done = False
    env_obs = env.reset()

    l1_mu, l1_std, l2_mu, l2_std = getStats(x_batch, w)

    l1_gamma = wdist.get_w('l1_gamma', w)
    l1_beta = wdist.get_w('l1_beta', w)
    l2_gamma = wdist.get_w('l2_gamma', w)
    l2_beta = wdist.get_w('l2_beta', w)

    batchstats = {"l1_mu" : l1_mu, "l1_std" : l1_std, "l2_mu" : l2_mu, "l2_std" : l2_std,
                  "l1_gamma" : l1_gamma,"l1_beta" : l1_beta,"l2_gamma" : l2_gamma,"l2_beta" : l2_beta}

    while not done:

        # Forward pass of policy
        _, _,l3 = forward(env_obs, w, batchstats=batchstats)

        # Step environment
        env_obs, rew, done, _ = env.step(l3)

        if animate:
            env.render()

        reward += rew

    return -reward

# Make environment
env = gym.make("DartHexapod-v1")
env.reset()
print("Action space: {}, observation space: {}".format(env.action_space.shape, env.observation_space.shape))
animate = True

N = env.action_space.shape[0]

x_batch = []

for i in range(32):
    obs, rew, done, _ = env.step(env.action_space.sample())
    x_batch.append(obs)

ctr = 0
while False:
    obs, _, _, _ = env.step(env.action_space.sample())
    if np.random.rand() < 0.05:
        env.reset()
    if np.random.rand() < 0.05:
        x_batch.append(obs)
        ctr += 1
    if ctr > 31:
        break

# Generate weights
wdist = Wdist()

afun = np.tanh
actfun = lambda x:x

print("afun: {}".format(afun))

wdist.addW((47, 8), 'w_l1')
wdist.addW((8,), 'b_l1')
wdist.addW((8,), 'l1_gamma')
wdist.addW((8,), 'l1_beta')

wdist.addW((8, 8), 'w_l2')
wdist.addW((8,), 'b_l2')

wdist.addW((8,), 'l2_gamma')
wdist.addW((8,), 'l2_beta')

wdist.addW((8, 18), 'w_l3')
wdist.addW((18,), 'b_l3')


N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
w = np.random.randn(N_weights) * 0.5

# Set initial values of gamma and beta
wdist.fill_w('l1_gamma', w, 1)
wdist.fill_w('l1_beta', w, 0)
wdist.fill_w('l2_gamma', w, 1)
wdist.fill_w('l2_beta', w, 0)

print("Comments: batchnorm")
es = cma.CMAEvolutionStrategy(w, 0.5)

try:
    es.optimize(f, iterations=2000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()

print(es.result.xbest, es.result.fbest, sep=',', file=open("hopper_weights.txt", "a"))


