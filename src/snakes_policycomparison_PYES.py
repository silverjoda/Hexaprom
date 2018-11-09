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

        self.fc1 = nn.Linear(18, 7)

    def forward(self, x):
        x = T.tanh(self.fc1(x))
        return x


class ConvPolicy(nn.Module):
    def __init__(self):
        super(ConvPolicy, self).__init__()

        self.fc_emb = nn.Linear(8, 4)
        self.fc_emb_2 = nn.Linear(4, 4)

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1)

        self.deconv1 = nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=3, stride=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=4, out_channels=1, kernel_size=3, stride=2)

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.deconv1.weight)
        nn.init.xavier_uniform_(self.deconv2.weight)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.afun = F.tanh

    def forward(self, x):
        obs = x[:, :4]
        j = x[:, 4:11]
        jd = x[:, 11:]
        jcat = T.cat([j.unsqueeze(1) ,jd.unsqueeze(1)], 1) # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv1(jcat))
        fm_c2 = self.afun(self.conv2(fm_c1))
        fm = fm_c2.squeeze(2)

        fc_emb = self.afun(self.fc_emb(T.cat((obs, fm), 1)))
        fc_emb = self.afun(self.fc_emb_2(fc_emb))
        fc_emb = fc_emb.unsqueeze(2)

        fm = self.afun(self.deconv1(fc_emb))
        fm = self.deconv2(T.cat((fm, fm_c1), 1))

        x = fm.unsqueeze(1)

        return x


class SymPolicy(nn.Module):
    def __init__(self):
        super(SymPolicy, self).__init__()

        self.fc_ji_1 = nn.Linear(2, 2)
        self.fc_ji_2_list = [nn.Linear(2, 4).double() for _ in range(7)]
        self.fc_e_1 = nn.Linear(8, 4)
        self.fc_e_2 = nn.Linear(4, 2)
        self.fc_jo_list = [nn.Linear(2, 1).double() for _ in range(7)]

        self.afun = F.tanh

    def forward(self, x):
        obs = x[:, :4]
        j = x[:, 4:11]
        jd = x[:, 11:]
        jcat = T.cat([j.unsqueeze(1), jd.unsqueeze(1)], 1)  # Concatenate j and jd so that they are 2 parallel channels

        j_in = T.zeros((1,4)).double()
        for i in range(7):
            j_in += self.fc_ji_2_list[i](self.afun(self.fc_ji_1(jcat[:, :, i])))
        j_in = self.afun(j_in)

        emb = self.afun(self.fc_e_1(T.cat([obs, j_in], 1)))
        emb = self.afun(self.fc_e_2(emb))

        j_out = []
        for i in range(7):
            j_out.append(self.fc_jo_list[i](emb + jcat[:, :, i]))

        j_out = T.cat(j_out, 1)

        return j_out


class RecPolicy(nn.Module):
    def __init__(self):
        super(RecPolicy, self).__init__()

        self.r_up = nn.RNNCell(2, 4)
        self.fc_obs_1 = nn.Linear(8, 4)
        self.fc_obs_2 = nn.Linear(4, 4)
        self.r_down = nn.RNNCell(4, 4)
        self.fc_out = nn.Linear(5, 1)

        self.afun = F.tanh


    def forward(self, x):
        obs = x[:, :4]
        j = x[:, 4:11]
        jd = x[:, 11:]
        jcat = T.cat([j.unsqueeze(1), jd.unsqueeze(1)], 1)  # Concatenate j and jd so that they are 2 parallel channels

        h = T.zeros(1, 4).double()

        h_up = []
        for i in reversed(range(7)):
            h = self.r_up(jcat[:, :, i], h)
            h_up.append(h)

        h_up.reverse()
        h = self.afun(self.fc_obs_2(self.afun(self.fc_obs_1(T.cat((obs, h), 1))))) #
        h_prev = h
        acts = []
        for i in range(7):
            h = self.r_down(h_up[i], h)
            acts.append(self.fc_out(T.cat((h_prev, j[:,i:i+1]),1)))
            h_prev = h
        return T.cat(acts, 1)


class TRecPolicy(nn.Module):
    def __init__(self):
        super(TRecPolicy, self).__init__()

        self.r_up = nn.GRUCell(2, 4)
        self.fc_obs_1 = nn.Linear(8, 4)
        self.fc_obs_2 = nn.Linear(4, 4)
        self.r_down = nn.GRUCell(4, 4)
        self.fc_out = nn.Linear(5, 1)

        self.afun = F.tanh

    def forward(self, x):
        obs = x[:, :4]
        j = x[:, 4:11]
        jd = x[:, 11:]
        jcat = T.cat([j.unsqueeze(1), jd.unsqueeze(1)], 1)  # Concatenate j and jd so that they are 2 parallel channels

        h = T.zeros(1, 4).double()

        h_up = []
        for i in reversed(range(7)):
            h = self.r_up(jcat[:, :, i], h)
            h_up.append(h)

        h = self.afun(self.fc_obs_2(self.afun(self.fc_obs_1(T.cat((obs, h), 1))))) #

        acts = []
        for i in range(7):
            h = self.r_down(h_up[i], h)
            acts.append(self.fc_out(T.cat((h, j[:,i:i+1]),1)))

        return T.cat(acts, 1)


class AggregPolicy(nn.Module):
    def __init__(self):
        super(AggregPolicy, self).__init__()

        self.fc_j_init = nn.Linear(2, 4)
        self.fc_m_init = nn.Linear(4, 4)
        self.rnn_j = nn.RNNCell(4, 4)
        self.rnn_m = nn.RNNCell(4, 4)
        self.fc_act = nn.Linear(4, 1)


    def forward(self, x):
        obs = x[:, :4]
        j = x[:, 4:11]
        jd = x[:, 11:]
        jcat = T.cat([j.unsqueeze(1), jd.unsqueeze(1)], 1)  # Concatenate j and jd so that they are 2 parallel channels

        h_j_list = [self.fc_j_init(jcat[:, :, i]) for i in range(7)]
        h_m = self.fc_m_init(obs)

        # 7 iterations
        for _ in range(7):
            h_m_new = self.rnn_m(h_j_list[0], h_m)

            h_j_new_list = [self.rnn_j(h_m + h_j_list[1], h_j_list[0])]
            # Go over each non-master joint
            for i in range(1, 6):
                h_j_new_list.append(self.rnn_j(h_j_list[i-1] + h_j_list[i+1], h_j_list[i]))
            h_j_new_list.append(self.rnn_j(h_j_list[5], h_j_list[6]))

            h_m = h_m_new
            h_j_list = h_j_new_list

        acts = T.cat([self.fc_act(h) for h in h_j_list], 1)

        return acts


def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        # Inject current parameters into policy ## CONSIDER MAKING HARD COPY OF POLICY HERE NOT TO INTERFERE WITH INITIAL POLICY ##
        pytorch_ES.vector_to_parameters(T.from_numpy(w), policy.parameters())

        while not done:

            # Remap the observations
            obs = np.concatenate((obs[0:1],obs[8:11],obs[1:8],obs[11:]))

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

    env_name, policyfun, iters, animate = params

    # Make environment
    env = gym.make(env_name)

    # Get environment dimensions
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]

    # Make policy
    policy = policyfun()

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

env_name = "SwimmerLong-v0"
#policyfunctions = [Baseline, ConvPolicy, SymPolicy, RecPolicy, AggregPolicy]
policyfunctions = [RecPolicy]

for p in policyfunctions:
    print("Training with {} policy.".format(p.__name__))
    fbest = train((env_name, p, 300, True))
    print("Policy {} max score: {}".format(p.__name__, fbest))

print("Done, exiting.")
exit()

