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
import os

class Baseline(nn.Module):
    def __init__(self, N):
        super(Baseline, self).__init__()
        self.N_links = int(N / 2)
        self.fc1 = nn.Linear(93, 40)

    def forward(self, x):
        x = self.fc1(x)
        return x


class ConvPolicy30(nn.Module):
    def __init__(self, N):
        super(ConvPolicy30, self).__init__()
        self.N_links = int(N / 2)

        # rep conv
        self.conv_1 = nn.Conv1d(12, 6, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv1d(6, 8, kernel_size=3, stride=1)
        self.conv_3 = nn.Conv1d(8, 8, kernel_size=3, stride=1)
        self.downsample = nn.AdaptiveAvgPool1d(5)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(10, 10, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(10, 10, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(10, 6, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_3 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_4 = nn.ConvTranspose1d(18, 6, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=13)

        self.afun = F.tanh

    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]

        # Get psi angle from observation quaternion
        _, _, psi = quaternion.as_euler_angles(np.quaternion(*(obs[0,3:7].numpy())))
        psi = T.tensor([psi], dtype=T.float32).unsqueeze(0)

        # (psi, psid)
        ext_obs = T.cat((psi, obsd[:, -1:]), 1)

        # Joints angles
        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((1, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(1, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((1, 6, -1))

        jcat = T.cat((jlrs, jdlrs), 1) # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(jcat))
        fm_c1_ds = self.downsample(fm_c1)
        fm_c2 = self.afun(self.conv_2(fm_c1_ds))
        fm_c3 = self.afun(self.conv_3(fm_c2))

        # Avg pool through link channels
        fm_links = self.pool(fm_c3) # (1, N, 1)

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_links, ext_obs.unsqueeze(2)),1)))
        emb_2 = self.afun(self.conv_emb_2(emb_1))

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(emb_2))
        fm_dc2 = self.afun(self.deconv_2(fm_dc1))
        fm_dc2_us = self.upsample(fm_dc2)
        fm_dc3 = self.afun(self.deconv_3(fm_dc2_us))
        fm_dc4 = self.deconv_4(T.cat((fm_dc3, jcat), 1))

        acts = fm_dc4.squeeze(2).view((1, -1))

        return acts[:, 2:]


class ConvPolicy14(nn.Module):
    def __init__(self, N):
        super(ConvPolicy14, self).__init__()
        self.N_links = int(N / 2)

        # rep conv
        self.conv_1 = nn.Conv1d(12, 4, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv1d(4, 3, kernel_size=3, stride=1, padding=0)

        self.downsample = nn.AdaptiveAvgPool1d(3)

        self.deconv_1 = nn.ConvTranspose1d(3, 4, kernel_size=3, stride=1, padding=0)
        self.deconv_2 = nn.ConvTranspose1d(8, 4, kernel_size=3, stride=1, padding=1)
        self.deconv_3 = nn.ConvTranspose1d(8, 4, kernel_size=3, stride=1, padding=1)
        self.deconv_4 = nn.ConvTranspose1d(16, 6, kernel_size=3, stride=1, padding=1)

        self.afun = T.tanh

    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]

        # Get psi angle from observation quaternion
        _, _, psi = quaternion.as_euler_angles(np.quaternion(*(obs[0,3:7].numpy())))
        psi = T.tensor([psi], dtype=T.float32).unsqueeze(0)

        # (psi, psid)
        ext_obs = T.cat((psi, obsd[:, 0:1], obsd[:, 5:6]), 1)

        # Joints angles
        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((1, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(1, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((1, 6, -1))

        jcat = T.cat((jlrs, jdlrs), 1) # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(jcat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c2_ds = self.downsample(fm_c2)
        fm_c3 = self.afun(self.conv_3(fm_c2_ds))
        fm_c4 = self.afun(self.conv_4(fm_c3))

        # Avg pool through link channels
        fm_comb = fm_c4 + ext_obs.unsqueeze(2)

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(fm_comb))
        fm_dc2 = self.afun(self.deconv_2(T.cat((fm_dc1, fm_c3), 1)))
        fm_dc2_us = F.interpolate(fm_dc2, size=7)
        fm_dc3 = self.afun(self.deconv_3(T.cat((fm_dc2_us, fm_c2), 1)))
        fm_dc4 = self.deconv_4(T.cat((fm_dc3, jcat), 1))

        acts = fm_dc4.squeeze(2).view((1, -1))

        return acts[:, 2:]


class ConvPolicy30x(nn.Module):
    def __init__(self, N):
        super(ConvPolicy30x, self).__init__()
        self.N_links = int(N / 2)

        # rep conv
        self.conv_1 = nn.Conv1d(12, 4, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv1d(4, 2, kernel_size=3, stride=1)
        self.conv_3 = nn.Conv1d(2, 1, kernel_size=3, stride=1)
        self.downsample = nn.AdaptiveAvgPool1d(5)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # TODO: Don't use downsampling, Leave the temporal features (however many there are left in the end). Avg pool
        # TODO: to combine with embedding and then recombine with conv feature on the way to action space

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(10, 10, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(10, 10, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(10, 6, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_3 = nn.ConvTranspose1d(6, 6, kernel_size=3, stride=1)
        self.deconv_4 = nn.ConvTranspose1d(18, 6, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=13)

        self.afun = F.tanh

    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]

        # Get psi angle from observation quaternion
        _, _, psi = quaternion.as_euler_angles(np.quaternion(*(obs[0,3:7].numpy())))
        psi = T.tensor([psi], dtype=T.float32).unsqueeze(0)

        # (psi, psid)
        ext_obs = T.cat((psi, obsd[:, -1:]), 1)

        # Joints angles
        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((1, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(1, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((1, 6, -1))

        jcat = T.cat((jlrs, jdlrs), 1) # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(jcat))
        fm_c1_ds = self.downsample(fm_c1)
        fm_c2 = self.afun(self.conv_2(fm_c1_ds))
        fm_c3 = self.afun(self.conv_3(fm_c2))

        # Avg pool through link channels
        fm_links = self.pool(fm_c3) # (1, N, 1)

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_links, ext_obs.unsqueeze(2)),1)))
        emb_2 = self.afun(self.conv_emb_2(emb_1))

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(emb_2))
        fm_dc2 = self.afun(self.deconv_2(fm_dc1))
        fm_dc2_us = self.upsample(fm_dc2)
        fm_dc3 = self.afun(self.deconv_3(fm_dc2_us))
        fm_dc4 = self.deconv_4(T.cat((fm_dc3, jcat), 1))

        acts = fm_dc4.squeeze(2).view((1, -1))

        return acts[:, 2:]


class ConvPolicy8(nn.Module):
    def __init__(self, N):
        super(ConvPolicy8, self).__init__()
        self.N_links = int(N / 2)

        # rep conv
        self.conv_1 = nn.Conv1d(12, 4, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(4, 8, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv1d(8, 8, kernel_size=3, stride=1)
        self.conv_4 = nn.Conv1d(8, 8, kernel_size=2, stride=1)

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(10, 8, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(8, 8, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(8, 4, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(4, 4, kernel_size=3, stride=1, padding=1)
        self.deconv_3 = nn.ConvTranspose1d(4, 8, kernel_size=3, stride=1, padding=1)
        self.deconv_4 = nn.ConvTranspose1d(14, 6, kernel_size=3, stride=1, padding=1)

        self.upsample = nn.Upsample(size=4)

        self.afun = F.tanh

    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]

        # Get psi angle from observation quaternion
        _, _, psi = quaternion.as_euler_angles(np.quaternion(*(obs[0,3:7].numpy())))
        psi = T.tensor([psi], dtype=T.float32).unsqueeze(0)

        # (psi, psid)
        ext_obs = T.cat((psi, obsd[:, -1:]), 1)

        # Joints angles
        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((1, 6, -1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(1, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((1, 6, -1))

        jcat = T.cat((jlrs, jdlrs), 1) # Concatenate j and jd so that they are 2 parallel channels

        fm_c1 = self.afun(self.conv_1(jcat))
        fm_c2 = self.afun(self.conv_2(fm_c1))
        fm_c3 = self.afun(self.conv_3(fm_c2))
        fm_c4 = self.afun(self.conv_4(fm_c3))

        # Combine obs with featuremaps
        emb_1 = self.afun(self.conv_emb_1(T.cat((fm_c4, ext_obs.unsqueeze(2)),1)))
        emb_2 = self.afun(self.conv_emb_2(emb_1))

        # Project back to action space
        fm_dc1 = self.afun(self.deconv_1(emb_2))
        fm_dc2 = self.afun(self.deconv_2(fm_dc1))
        fm_dc3 = self.afun(self.deconv_3(fm_dc2))
        fm_upsampled = self.upsample(fm_dc3)
        fm_dc4 = self.deconv_4(T.cat((fm_upsampled, jlrs), 1))

        acts = fm_dc4.squeeze(2).view((1, -1))

        return acts[:, 2:]


class RecPolicy(nn.Module):
    def __init__(self, N):
        super(RecPolicy, self).__init__()

        # Amount of cells that the centipede has
        self.N_links = int(N / 2)

        # Cell RNN hidden
        self.n_hidden = 8

        # RNN for upwards pass
        self.r_up = nn.RNNCell(12, self.n_hidden)

        # Global obs
        self.fc_obs_1 = nn.Linear(13, self.n_hidden)
        self.fc_obs_2 = nn.Linear(self.n_hidden, self.n_hidden)

        # RNN for backwards pass
        self.r_down = nn.RNNCell(self.n_hidden, self.n_hidden)

        # From hidden to cell actions
        self.cell_unfc1 = nn.Linear(self.n_hidden * 2, 6)

        # Last conv layer to join with local observations
        #self.unconv_act = nn.Conv1d(3, 1, 1)

        self.afun = T.tanh


    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]
        obs_cat = T.cat((obs, obsd), 1)

        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]),1)
        jdl = T.cat((T.zeros(1, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]),1)

        h = T.zeros(1, self.n_hidden)

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
            shift = 6 * i
            j = jl[:, shift:shift + 6]
            jd = jdl[:, shift:shift + 6]
            jcat = T.cat((j.unsqueeze(1),jd.unsqueeze(1)), 1)


            # act_h = self.cell_unfc1(T.cat((h, h_up[i]), 1))
            # act_cat = T.cat((jcat, act_h.unsqueeze(1)), 1)
            # act_final = self.unconv_act(act_cat).squeeze(1)

            act_final = self.cell_unfc1(T.cat((h, h_up[i]), 1))
            acts.append(act_final)
            h = self.r_down(h_up[i], h)

        return T.cat(acts, 1)[:, 2:]


class StatePolicy(nn.Module):
    def __init__(self, N):
        super(StatePolicy, self).__init__()
        self.N_links = int(N / 2)

        # Rep conv
        self.conv_1 = nn.Conv1d(7, 7, kernel_size=3, stride=1, padding=1)

        # Obs to state
        self.comp_mat = nn.Parameter(T.randn(1, 7, 1, 3))

        # State to action
        self.act_mat = nn.Parameter(T.randn(1, 6, 1, 2))

        # States
        self.reset()

        self.afun = T.tanh

    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]

        # Get psi angle from observation quaternion
        _, _, psi = quaternion.as_euler_angles(np.quaternion(*(obs[0,3:7].numpy())))
        psi = T.tensor([psi], dtype=T.float32).unsqueeze(0)

        # (psi, psid)
        ext_rs = T.cat((psi.view(1,1,1,1), obsd[:, 0:1].view(1,1,1,1)), 3).repeat(1,1,self.N_links,1)

        # Joints angles
        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((1, 6, self.N_links, 1))

        # Joint angle velocities
        jdl = T.cat((T.zeros(1, 2), x[:, 7 + self.N_links * 6 - 2 + 6:]), 1)
        jdlrs = jdl.view((1, 6, self.N_links, 1))

        obscat = T.cat((T.cat((jlrs, jdlrs), 3), ext_rs), 1) # Concatenate j and jd so that they are 2 parallel channels

        comp_mat_full = self.comp_mat.repeat(1,1,self.N_links,1)
        states = self.states
        for i in range(3):
            # Concatenate observations with states
            x = T.cat((obscat, states), 3)

            # Multiply elementwise through last layer to get prestate map
            x = self.afun((x * comp_mat_full).sum(3))

            # Convolve prestate map to get new states
            states = self.afun(self.conv_1(x).unsqueeze(3))

        # Turn states into actions
        acts = self.act_mat.repeat(1,1,self.N_links,1) * T.cat((states[:,:6,:,:], jdlrs), 3)
        acts = acts.sum(3).view((1, -1))

        return acts[:, 2:]

    def reset(self):
        self.states = T.randn(1, 7, self.N_links, 1)


class PhasePolicy(nn.Module):
    def __init__(self, N):
        super(PhasePolicy, self).__init__()
        self.N_links = int(N / 2)

        # Set phase states
        self.reset()

        # Increment matrix which will be added to phases every step
        self.step_increment = T.ones(1, 6, self.N_links) * 0.01

        self.conv_obs = nn.Conv1d(7, 6, kernel_size=3, stride=1, padding=1)
        self.conv_phase = nn.Conv1d(6, 6, kernel_size=3, stride=1, padding=1)

        self.afun = T.tanh


    def step_phase(self):
        self.phases = T.fmod(self.phases + self.step_increment, np.pi)


    def modify_phase(self, mask):
        self.phases = T.fmod(self.phases + mask, np.pi)


    def reset(self):
        self.phases = T.randn(1, 6, self.N_links) * 0.01


    def forward(self, x):
        obs = x[:, :7]

        # Get psi angle from observation quaternion
        _, _, psi = quaternion.as_euler_angles(np.quaternion(*(obs[0,3:7].numpy())))
        psi = T.tensor([psi], dtype=T.float32).unsqueeze(0)

        # (psi, psid)
        ext_rs = psi.view(1,1,1).repeat(1,1,self.N_links)

        # Joints angles
        jl = T.cat((T.zeros(1, 2), x[:, 7:7 + self.N_links * 6 - 2]), 1)
        jlrs = jl.view((1, 6, self.N_links))

        obscat = T.cat((jlrs, ext_rs), 1) # Concatenate j and jd so that they are 2 parallel channels

        phase_fm = self.afun(self.conv_obs(obscat))
        phase_deltas = self.afun(self.conv_phase(phase_fm))

        self.modify_phase(phase_deltas)
        self.step_phase()

        # Phases directly translate into torques
        #acts = self.phases.view(1,-1) - (np.pi / 2)

        # Phases are desired angles
        acts = (((self.phases - (np.pi / 2)) - jlrs) * 0.1).view(1,-1)


        return acts[:, 2:]



def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()
        policy.reset()

        pytorch_ES.vector_to_parameters(T.from_numpy(w.astype(np.float32)), policy.parameters())

        step_ctr = 0
        step_ctr_lim = 300

        while not done:

            # Get action from policy
            with T.no_grad():
                act = policy(T.from_numpy(np.expand_dims(obs.astype(np.float32), 0)))[0].numpy()

            # Step environment
            obs, rew, done, _ = env.step(act)

            if animate:
                env.render()

            reward += rew

            step_ctr += 1
            if step_ctr > step_ctr_lim:
                done = True

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

    pytorch_ES.vector_to_parameters(T.from_numpy(es.result.xbest.astype(np.float32)), policy.parameters())

    return -es.result.fbest, policy

N = 8
env_name = "Centipede{}-v0".format(N)
env = gym.make(env_name)

print(env.observation_space, env.action_space)
#policyfunctions = [Baseline, ConvPolicy, SymPolicy, RecPolicy, AggregPolicy]
policyfunctions = [PhasePolicy]

#===========
# M = 100
# act = env.action_space.sample()
# obs = env.reset()
# p = Baseline()
# import time
# t1 = time.clock()
# for i in range(M):
#     #env.step(act)
#     p(T.FloatTensor(obs).unsqueeze(0))
# t2 = time.clock()
#
# print((t2-t1)/M)
# exit()
# #===========

for p in policyfunctions:
    print("Training with {} policy.".format(p.__name__))
    fbest, policy = train((env_name, p(N).float(), 10000, True))
    print("Policy {} max score: {}".format(p.__name__, fbest))
    ctr = 0
    while os.path.exists("agents/ES/{}_{}.p".format(p.__name__, ctr)):
        ctr += 1
    T.save(policy, "agents/ES/{}_{}.p".format(p.__name__, ctr))

exit()

# Evaluate
policy = T.load("agents/ES/ConvPolicy8_0.p")
f = f_wrapper(env, policy, True)
for i in range(10):
    r = f(pytorch_ES.parameters_to_vector(policy.parameters()).detach().numpy())
    print(r)

print("Done, exiting.")
exit()

