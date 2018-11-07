import numpy as np
import gym
import cma
from time import sleep
import quaternion
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ES


class Baseline(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Baseline, self).__init__()

        self.fc1 = nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return x


class ConvPolicy(nn.Module):
    def __init__(self):
        super(ConvPolicy, self).__init__()

        self.fc_obs = nn.Linear(4, 2)
        self.fc_emb = nn.Linear(2, 2)

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1)

        self.deconv1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=2, stride=1)
        self.deconv2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=3, stride=2)

    def forward(self, x):
        obs = x[:, :4]
        j = x[:, 4:11]
        jd = x[:, 11:]
        jcat = torch.cat([j,jd], 1).unsqueeze(1) # Concatenate j and jd so that they are 2 parallel channels

        fc_obs = F.relu(self.fc_obs(obs))

        fm = F.relu(self.conv1(jcat))
        fm = F.relu(self.conv2(fm))
        fm = fm.squeeze(1)

        fc_emb = F.relu(self.fc_emb(fc_obs + fm))
        fc_emb = fc_emb.unsqueeze(1)

        fm = F.relu(self.deconv1(fc_emb))
        fm = self.deconv2(fm)

        x = fm

        return x


class SymPolicy(nn.Module):
    def __init__(self):
        super(SymPolicy, self).__init__()

        self.fc_ji = nn.Linear(2, 2)
        self.fc_e = nn.Linear(6, 2)
        self.fc_jo = nn.Linear(2, 1)


    def forward(self, x):
        obs = x[:, :5]
        j = x[:, 5:12]
        jd = x[:, 12:]
        jcat = torch.cat([j,jd], 1).unsqueeze(1) # Concatenate j and jd so that they are 2 parallel channels

        j_in = torch.zeros((1,2))
        for i in range(7):
            j_in += self.fc_j1(jcat[:, i])

        j_in = F.relu(j_in)
        emb = F.relu(self.fc_e(torch.cat([obs, j_in], 1)))

        j_out = []
        for i in range(7):
            j_out.append(self.fc_jo(emb + jcat[i]))

        j_out = torch.stack(j_out, 1)

        return j_out


class RecPolicy(nn.Module):
    def __init__(self):
        super(RecPolicy, self).__init__()

        self.r_up = nn.GRUCell(2, 2)
        self.fc_obs = nn.Linear(6, 2)
        self.r_down = nn.GRUCell(2, 2)
        self.fc_out = nn.Linear(2, 1)

    def forward(self, x):
        obs = x[:, :5]
        j = x[:, 5:12]
        jd = x[:, 12:]
        jcat = torch.cat([j,jd], 1).unsqueeze(1) # Concatenate j and jd so that they are 2 parallel channels

        h = torch.zeros(1, 2).float()

        h_up = []
        for i in range(7):
            h = self.rnn(jcat[i], h)
            h_up.append(h)

        h = self.fc_obs(torch.cat((obs, h), 1))

        acts = []
        for i in range(7):
            h = self.rnn(h_up[i], h)
            acts.append(self.fc_out(h))

        return torch.stack(acts, 1)


class AggregPolicy(nn.Module):
    def __init__(self):
        super(AggregPolicy, self).__init__()

        self.fc_j_init = nn.Linear(2, 4)
        self.fc_m_init = nn.Linear(4, 4)
        self.rnn_j = nn.GRUCell(4, 4)
        self.rnn_m = nn.GRUCell(4, 4)
        self.fc_act = nn.Linear(4, 1)


    def forward(self, x):
        obs = x[:, :5]
        j = x[:, 5:12]
        jd = x[:, 12:]
        jcat = torch.cat([j,jd], 1).unsqueeze(1) # Concatenate j and jd so that they are 2 parallel channels

        h_j_list = [self.fc_init(jcat[:, i]) for i in range(7)]
        h_m = self.fc_m_init(obs)

        # 7 iterations
        for _ in range(7):
            h_m_new = self.rnn_m(h_j_list[0], h_m)

            h_j_new_list = [self.rnn_j(torch.cat((h_m, h_j_list[1]), 1), h_j_list[0])]
            # Go over each non-master joint
            for i in range(1,6):
                h_j_new_list.append(self.rnn_j(torch.cat((h_j_list[i-1], h_j_list[i+1]), 1), h_j_list[i]))
            h_j_new_list.append(self.rnn_j(h_j_list[5], h_j_list[-1]))

            h_m = h_m_new
            h_j_list = h_j_new_list

        acts = torch.stack([self.fc_act(h for h in h_j_list)], 1)

        return acts

def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        # Inject current parameters into policy ## CONSIDER MAKING HARD COPY OF POLICY HERE NOT TO INTERFERE WITH INITIAL POLICY ##
        pytorch_ES.vector_to_parameters(torch.from_numpy(w), policy.parameters())

        while not done:

            # Remap the observations
            obs = np.concatenate((obs[0:1],obs[8:11],obs[1:8],obs[11:]))

            # Get action from policy
            with torch.no_grad():
                act = policy(torch.from_numpy(np.expand_dims(obs, 0)))[0].numpy()

            # Step environment
            obs, rew, done, _ = env.step(act)

            if animate:
                env.render()

            reward += rew

        return -reward
    return f


def train(params):

    env_name, iters, n_hidden, animate = params

    # Make environment
    env = gym.make(env_name)

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    policy = Baseline(obs_dim, act_dim)
    w = pytorch_ES.parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, 0.5)
    f = f_wrapper(env, policy, animate)

    print("Env: {} Action space: {}, observation space: {}, N_params: {}, comments: ...".format(env_name, env.action_space.shape,
                                                                                  env.observation_space.shape, len(w)))

    try:
        while not es.stop():
            X = es.ask()
            es.tell(X, [f(x) for x in X])
            es.disp()
    except KeyboardInterrupt:
        print("User interrupted process.")

    return es.result.fbest

env_name = "SwimmerLong-v0"
train((env_name, 1000, 7, True))
exit()

