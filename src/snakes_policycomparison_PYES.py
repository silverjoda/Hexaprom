import numpy as np
import gym
import cma
from time import sleep
import quaternion
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ES


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()

        self.fc1 = nn.Linear(18, 7)

    def forward(self, x):
        x = nn.tanh(self.fc1(x))
        return x


class ConvPolicy(nn.Module):
    def __init__(self):
        super(ConvPolicy, self).__init__()

        self.fc_emb = nn.Linear(6, 2)

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1)

        self.deconv1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=2, stride=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
        self.deconv3 = nn.ConvTranspose1d(in_channels=2, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
        obs = x[:, :4]
        j = x[:, 4:11]
        jd = x[:, 11:]
        jcat = T.cat([j.unsqueeze(1) ,jd.unsqueeze(1)], 1) # Concatenate j and jd so that they are 2 parallel channels

        fm = T.tanh(self.conv1(jcat))
        fm = T.tanh(self.conv2(fm))
        fm = fm.squeeze(1)

        fc_emb = T.tanh(self.fc_emb(T.cat((obs, fm), 1)))
        fc_emb = fc_emb.unsqueeze(1)

        fm = T.tanh(self.deconv1(fc_emb))
        fm = self.deconv2(fm)
        fm = self.deconv3(T.cat((fm, j.unsqueeze(1)), 1))

        x = fm.unsqueeze(1)

        return x


class SymPolicy(nn.Module):
    def __init__(self):
        super(SymPolicy, self).__init__()

        self.fc_ji = nn.Linear(2, 2)
        self.fc_e = nn.Linear(6, 2)
        self.fc_jo = nn.Linear(2, 1)


    def forward(self, x):
        obs = x[:, :4]
        j = x[:, 4:11]
        jd = x[:, 11:]
        jcat = T.cat([j.unsqueeze(1), jd.unsqueeze(1)], 1)  # Concatenate j and jd so that they are 2 parallel channels

        j_in = T.zeros((1,2)).double()
        for i in range(7):
            j_in += self.fc_ji(jcat[:, :, i])

        j_in = T.tanh(j_in)
        emb = T.tanh(self.fc_e(T.cat([obs, j_in], 1)))

        j_out = []
        for i in range(7):
            j_out.append(self.fc_jo(emb + jcat[:, :, i]))

        j_out = T.cat(j_out, 1)

        return j_out


class RecPolicy(nn.Module):
    def __init__(self):
        super(RecPolicy, self).__init__()

        self.r_up = nn.GRUCell(2, 2)
        self.fc_obs = nn.Linear(6, 2)
        self.r_down = nn.GRUCell(2, 2)
        self.fc_out = nn.Linear(2, 1)

    def forward(self, x):
        obs = x[:, :4]
        j = x[:, 4:11]
        jd = x[:, 11:]
        jcat = T.cat([j.unsqueeze(1), jd.unsqueeze(1)], 1)  # Concatenate j and jd so that they are 2 parallel channels

        h = T.zeros(1, 2).double()

        h_up = []
        for i in reversed(range(7)):
            h = self.r_up(jcat[:, :, i], h)
            h_up.append(h)

        h = self.fc_obs(T.cat((obs, h), 1))

        acts = []
        for i in range(7):
            h = self.r_down(h_up[i], h)
            acts.append(self.fc_out(h))

        return T.cat(acts, 1)


class AggregPolicy(nn.Module):
    def __init__(self):
        super(AggregPolicy, self).__init__()

        self.fc_j_init = nn.Linear(2, 4)
        self.fc_m_init = nn.Linear(4, 4)
        self.rnn_j = nn.GRUCell(4, 4)
        self.rnn_m = nn.GRUCell(4, 4)
        self.fc_act = nn.Linear(4, 1)


    def forward(self, x):
        obs = x[:, :4]
        j = x[:, 4:11]
        jd = x[:, 11:]
        jcat = T.cat([j.unsqueeze(1), jd.unsqueeze(1)], 1)  # Concatenate j and jd so that they are 2 parallel channels

        h_j_list = [self.fc_j_init(jcat[:, i]) for i in range(7)]
        h_m = self.fc_m_init(obs)

        # 7 iterations
        for _ in range(7):
            h_m_new = self.rnn_m(h_j_list[0], h_m)

            h_j_new_list = [self.rnn_j(T.cat((h_m, h_j_list[1]), 1), h_j_list[0])]
            # Go over each non-master joint
            for i in range(1,6):
                h_j_new_list.append(self.rnn_j(T.cat((h_j_list[i-1], h_j_list[i+1]), 1), h_j_list[i]))
            h_j_new_list.append(self.rnn_j(h_j_list[5], h_j_list[-1]))

            h_m = h_m_new
            h_j_list = h_j_new_list

        acts = T.stack([self.fc_act(h for h in h_j_list)], 1)

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
policyfunctions = [AggregPolicy]

for p in policyfunctions:
    print("Training with {} policy.".format(p.__name__))
    fbest = train((env_name, p, 100, True))
    print("Policy {} max score: {}".format(p.__name__, fbest))

print("Done, exiting.")
exit()

