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


class ConvPolicy30(nn.Module):
    def __init__(self, N):
        super(ConvPolicy30, self).__init__()
        self.N_links = int(N / 2)

        # rep conv
        self.conv_1 = nn.Conv1d(12, 6, kernel_size=3, stride=1)
        self.conv_2 = nn.Conv1d(6, 4, kernel_size=3, stride=1)
        self.conv_3 = nn.Conv1d(4, 4, kernel_size=3, stride=1)
        self.downsample = nn.AdaptiveAvgPool1d(5)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Embedding layers
        self.conv_emb_1 = nn.Conv1d(6, 4, kernel_size=1, stride=1)
        self.conv_emb_2 = nn.Conv1d(4, 4, kernel_size=1, stride=1)

        self.deconv_1 = nn.ConvTranspose1d(4, 4, kernel_size=3, stride=1)
        self.deconv_2 = nn.ConvTranspose1d(4, 2, kernel_size=3, stride=1)
        self.deconv_3 = nn.ConvTranspose1d(2, 2, kernel_size=3, stride=1)
        self.deconv_4 = nn.ConvTranspose1d(14, 6, kernel_size=3, stride=1, padding=1)
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

        # Amount of cells that the centipede has
        self.N_links = int(N / 2)

        # State dim
        self.s_dim = 3

        # Create states
        self.reset()

        # Master filter
        self.f_m =  nn.Linear(5, self.s_dim)

        # Rep filters
        self.f_1 = nn.Linear(1 * self.s_dim + 2, self.s_dim)
        self.f_6 = nn.Linear(4 * self.s_dim + 2, self.s_dim)
        self.f_8 = nn.Linear(6 * self.s_dim + 2, self.s_dim)

        self.afun = T.tanh


    def forward(self, x):
        obs = x[:, :7]
        obsd = x[:, 7 + self.N_links * 6 - 2: 7 + self.N_links * 6 - 2 + 6]
        obs_cat = T.cat((obs, obsd), 1)

        obs_proc = None

        jl = T.cat((T.zeros(1, 2).double(), x[:, 7:7 + self.N_links * 6 - 2]),1)
        jdl = T.cat((T.zeros(1, 2).double(), x[:, 7 + self.N_links * 6 - 2 + 6:]),1)

        self.m_s_new = self.f_m(T.cat(obs_proc, jl[0:1], jl[2:3], jl[4:5], jl[5:6]), 1)

        # Vertebra lr
        self.vert_list_new = []

        # Left hip
        self.hip_l_list_new = []

        # Right hip
        self.hip_r_list_new = []

        # Left foot
        self.foot_l_list_new = []

        # Right foot
        self.foot_r_list_new = []

        for i in range(self.N_links):
            shift = 6 * i
            j = jl[:, shift:shift + 6]

            self.vert_list_new.append(self.f_8(T.cat(j[:, 0:1], None), 1))

        # States
        self.m_s = self.m_s_new

        # Vertebra lr
        self.vert_list = self.vert_list_new

        # Left hip
        self.hip_l_list = self.hip_l_list_new

        # Right hip
        self.hip_r_list = self.hip_r_list_new

        # Left foot
        self.foot_l_list = self.foot_l_list_new

        # Right foot
        self.foot_r_list = self.foot_r_list_new


        acts = None

        return acts


    def reset(self):

        # States
        self.m_s = T.zeros(1, self.s_dim)

        # Vertebra lr
        self.vert_list = [T.zeros(1, self.s_dim) for _ in range(self.N_links)]

        # Left hip
        self.hip_l_list = [T.zeros(1, self.s_dim) for _ in range(self.N_links)]

        # Right hip
        self.hip_r_list = [T.zeros(1, self.s_dim) for _ in range(self.N_links)]

        # Left foot
        self.foot_l_list = [T.zeros(1, self.s_dim) for _ in range(self.N_links)]

        # Right foot
        self.foot_r_list = [T.zeros(1, self.s_dim) for _ in range(self.N_links)]


def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        # Inject current parameters into policy ## CONSIDER MAKING HARD COPY OF POLICY HERE NOT TO INTERFERE WITH INITIAL POLICY ##
        pytorch_ES.vector_to_parameters(T.from_numpy(w.astype(np.float32)), policy.parameters())

        step_ctr = 0
        step_ctr_lim = 700

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
                break

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

N = 8
env_name = "Centipede{}-v0".format(N)
env = gym.make(env_name)
print(env.observation_space, env.action_space)
#policyfunctions = [Baseline, ConvPolicy, SymPolicy, RecPolicy, AggregPolicy]
policyfunctions = [ConvPolicy8]

# NOTE: GEAR REDUCED TO 40 IN XML

for p in policyfunctions:
    print("Training with {} policy.".format(p.__name__))
    fbest, policy = train((env_name, p(N).float(), 10, False))
    print("Policy {} max score: {}".format(p.__name__, fbest))
    T.save(policy, "agents/ES/{}.p".format(p.__name__))


print("Done, exiting.")
exit()

