import numpy as np
import gym
import cma
from time import sleep
import quaternion
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pytorch_ES


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(309, 148)

    def forward(self, x):
        x = T.tanh(self.fc1(x))
        return x


class ConvPolicy(nn.Module):
    def __init__(self, N):
        super(ConvPolicy, self).__init__()
        self.N_cells = int(N / 2)

        # cell fc
        self.n_cell_channels = 4
        self.cell_fc1 = nn.Linear(8, self.n_cell_channels)
        self.cell_unfc1 = nn.Linear(self.n_cell_channels, 8)

        # rep conv
        self.conv_1 = nn.Conv1d(self.n_cell_channels, 2, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv1d(self.n_cell_channels, 4, kernel_size=3, stride=1)
        self.conv_3 = nn.Conv1d(self.n_cell_channels, 4, kernel_size=3, stride=1)
        self.downsample = nn.AvgPool1d(2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Embedding layers
        self.fc1 = nn.Linear(6, 4)
        self.fc2 = nn.Linear(4, 4)

        self.deconv_1 = nn.ConvTranspose1d(4, 4, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(4, 2, kernel_size=3, stride=1)
        self.deconv_3 = nn.ConvTranspose1d(2, 1, kernel_size=3, stride=1)
        self.upsample = nn.Upsample(scale_factor=2)

        self.afun = F.tanh

    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_cells * 4: 7 + self.N_cells * 4 + 4]
        obs_cat = T.cat((obs, obsd), 1)
        j = x[:, 7:7 + self.N_cells * 4]
        jd = x[:, -(7 + self.N_cells * 4):]
        jcat = T.cat([j.unsqueeze(1) ,jd.unsqueeze(1)], 1) # Concatenate j and jd so that they are 2 parallel channels



        fm_c1 = self.afun(self.conv1(jcat))
        fm_c2 = self.afun(self.conv2(fm_c1))
        fm = fm_c2.squeeze(2)

        fc_emb = self.afun(self.fc_emb(T.cat((obs, fm), 1)))
        fc_emb = self.afun(self.fc_emb_2(fc_emb))
        fc_emb = fc_emb.unsqueeze(2)

        fm = self.afun(self.deconv1(fc_emb))
        fm = self.deconv2(T.cat((fm, fm_c1), 1))
        fm = self.deconv3(T.cat((fm, j.unsqueeze(1)), 1))

        x = fm.unsqueeze(1)

        return x


class RecPolicy(nn.Module):
    def __init__(self, N):
        super(RecPolicy, self).__init__()

        # Amount of cells that the centipede has
        self.N_links = int(N / 2)

        # Cell RNN hidden
        self.n_hidden = 8

        # From hidden to cell actions
        self.cell_unfc1 = nn.Linear(self.n_hidden * 2 + 2, 6)

        # RNN for upwards pass
        self.r_up = nn.RNNCell(12, self.n_hidden)

        # Global obs
        self.fc_obs_1 = nn.Linear(13, self.n_hidden)
        self.fc_obs_2 = nn.Linear(self.n_hidden, self.n_hidden)

        # RNN for backwards pass
        self.r_down = nn.RNNCell(self.n_hidden, self.n_hidden)

        self.afun = T.tanh


    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]
        obs_cat = T.cat((obs, obsd), 1)
        jl = T.cat((T.zeros(1, 2).double(),x[:, 7:7 + self.N_links * 6 - 2]))
        jdl = T.cat((T.zeros(1, 2).double(), x[:, 7 + self.N_links * 6 - 2 + 6:]))

        h = T.zeros(1, self.n_hidden).double()

        h_up = []
        for i in reversed(range(self.N_links)):
            h_up.append(h)
            shift = 6 * i
            j = jl[:, shift:shift + 6]
            jd = jdl[:, shift:shift + 6]
            local_c = T.cat((j, jd), 1)
            h = self.r_up(local_c, h)

        h_up.reverse()
        h = self.afun(self.fc_obs_2(self.afun(self.fc_obs_1(obs_cat))))

        acts = []
        for i in range(self.N_links):
            acts.append(self.cell_unfc1(T.cat((h, h_up[i]), 1)))
            h = self.r_down(h_up[i], h)

        return T.cat(acts, 1)[:, 2:]


def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        # Inject current parameters into policy ## CONSIDER MAKING HARD COPY OF POLICY HERE NOT TO INTERFERE WITH INITIAL POLICY ##
        pytorch_ES.vector_to_parameters(T.from_numpy(w), policy.parameters())

        while not done:

            # Get action from policy
            with T.no_grad():
                act = policy(T.from_numpy(np.expand_dims(obs, 0)))[0].numpy()

            # Step environment
            obs, rew, done, _ = env.step(act)

            if animate:
                env.render()

            reward += rew

        return -reward
    return f


def train(params):

    env_name, policy, iters, animate = params

    # Make environment
    env = gym.make(env_name)

    # Get environment dimensions
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    # Make initial weight vectors from policy
    w = pytorch_ES.parameters_to_vector(policy.parameters()).detach().numpy()

    # Make optimization objective
    es = cma.CMAEvolutionStrategy(w, 0.5)

    # Make criterial function
    f = f_wrapper(env, policy, animate)

    # Print information
    print("Env: {} Action space: {}, observation space: {}, N_params: {}, comments: ...".format(env_name, env.action_space.shape,
                                                                                  env.observation_space.shape, len(w)))

    # Optimize
    it = 0
    try:
        while not es.stop():
            X = es.ask()
            es.tell(X, [f(x) for x in X])
            es.disp()

            if it > iters:
                break
            else:
                it += 1

    except KeyboardInterrupt:
        print("User interrupted process.")

    return -es.result.fbest

N = 50
env_name = "Centipede{}-v0".format(N)
env = gym.make(env_name)
print(env.observation_space, env.action_space)
exit()
#policyfunctions = [Baseline, ConvPolicy, SymPolicy, RecPolicy, AggregPolicy]
policyfunctions = [RecPolicy]

for p in policyfunctions:
    print("Training with {} policy.".format(p.__name__))
    fbest = train((env_name, p(N), 300, True))
    print("Policy {} max score: {}".format(p.__name__, fbest))

print("Done, exiting.")
exit()

