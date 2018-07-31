#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from policy import Policy
from value_function import NNValueFunction
from predictor import Predictor
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
import random


class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_gym(env_name):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler, animate=False):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    scale, offset = scaler.get()

    while not done:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        #print(action)
        actions.append(action)
        obs, reward, done, _ = env.step(np.squeeze(action, axis=0))
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)


    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, episodes, animate):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler, animate)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })


def eval_agent(env, policy, logger, obs_dim, act_dim, num_episodes):

    policy.restore_weights()

    scaler = Scaler(obs_dim)

    print("Restored weights, evaluating...")

    for i_episode in range(num_episodes):
        run_episode(env, policy, scaler, True)

    env.kill()


def make_predictor_dataset(trajectories):

    # Make list of all [obs,act,n_obs]
    obss, acts, rews, n_obss = [], [], [], []
    for traj in trajectories:
        action_list = traj['actions']
        reward_list = traj['rewards']
        observation_list = traj['unscaled_obs']

        # Add to dataset lists
        obss.extend(observation_list[:-1])
        acts.extend(action_list[:-1])
        rews.extend(reward_list[:-1])
        n_obss.extend(observation_list[1:])

    # Shuffle list randomly to decorrelate states
    c = list(zip(obss, acts, rews, n_obss))
    random.shuffle(c)

    return zip(*c)


def main(env_name, num_episodes, gamma, lam, kl_targ, batch_size, hid1_mult, policy_logvar, animate, evaluate, load_ckpt):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
    """

    killer = GracefulKiller()
    env, obs_dim, act_dim = init_gym(env_name)
    now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
    logger = Logger(logname=env_name, now=now)
    scaler = Scaler(obs_dim)
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar, random=True)
    predictor = Predictor(obs_dim, act_dim)

    if evaluate:
        print("Evaluating: ")
        eval_agent(env, policy, predictor, logger, obs_dim, act_dim)
        exit()

    if load_ckpt:
        print("Loading last ckpt: ")
        predictor.restore_weights()

    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, logger, 5, animate)
    episode = 0

    # Error lists
    state_errveclist = []
    rew_errlist = []

    moving_avg = 0
    h = 100

    while episode < num_episodes:
        trajectories = run_policy(env, policy, scaler, logger, batch_size, animate)

        # Train predictor
        pbs = 64
        obs_dataset, act_dataset, rew_dataset, n_obs_dataset = make_predictor_dataset(trajectories)
        for i in range(len(obs_dataset) // pbs - 1):
            obs = obs_dataset[i * pbs : i * pbs + pbs]
            act = act_dataset[i * pbs : i * pbs + pbs]
            rew = rew_dataset[i * pbs : i * pbs + pbs]
            n_obs = n_obs_dataset[i * pbs: i * pbs + pbs]
            state_err, rew_err = predictor.train(obs, act, rew, n_obs)
            state_errveclist.append(state_err)
            rew_errlist.append(rew_err)

            n_examples = len(state_err)

            q_total_err, z_err, quat_err, joint_err, v_total_err, xyzvel_err, anglevel_err, joint_vel_err = 0,0,0,0,0,0,0,0
            for j in range(n_examples):
                q_total_err += np.mean(state_err[j][:24])
                z_err += state_err[j][0]
                quat_err += np.mean(state_err[j][1:5])
                joint_err += np.mean(state_err[j][5:24])

                v_total_err += np.mean(state_err[j][24:])
                xyzvel_err += np.mean(state_err[j][24:27])
                anglevel_err += np.mean(state_err[j][27:30])
                joint_vel_err += np.mean(state_err[j][30:47])

            q_total_err /= n_examples
            z_err /= n_examples
            quat_err /= n_examples
            joint_err /= n_examples
            v_total_err /= n_examples
            xyzvel_err /= n_examples
            anglevel_err /= n_examples
            joint_vel_err /= n_examples

            print("Batch {}/{}, Total state error: {}, Total rew error: {}, Q_total_err: {}, z_err: {}, quat_err: {}, joint_err: {}, v_total_err: {}, xyzvel_err: {}, anglevel_err: {}, joint_vel_err: {}".
                  format(episode + i,num_episodes,np.mean([np.mean(s) for s in state_err]),np.mean([np.mean(r) for r in rew_err]),
                         q_total_err, z_err, quat_err, joint_err, v_total_err, xyzvel_err, anglevel_err, joint_vel_err))

            # total_err = np.mean([np.mean(s) for s in state_err])
            # total_rew_err = np.mean([np.mean(r) for r in rew_err])
            #
            # moving_avg += (total_err / h - moving_avg / h)
            #
            # print("Batch {}/{}, Total state error: {}, Moving avg: {}, Total rew error: {},".
            #       format(episode + i, num_episodes, total_err, moving_avg, total_rew_err))

        episode += len(trajectories)

        if killer.kill_now:
            if input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False

    predictor.save_weights()
    print("Saved predictor weights")
    logger.close()
    policy.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('env_name', type=str, help='OpenAI Gym environment name')
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=1000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=10)
    parser.add_argument('-m', '--hid1_mult', type=int,
                        help='Size of first hidden layer for value and policy NNs'
                             '(integer multiplier of observation dimension)',
                        default=10)
    parser.add_argument('-v', '--policy_logvar', type=float,
                        help='Initial policy log-variance (natural log of variance)',
                        default=-1.0)
    parser.add_argument('-a', '--animate', action='store_true', default=False, help="Render scene.")   
    parser.add_argument('-r', '--evaluate', action='store_true', default=False, help="Evaluate.")
    parser.add_argument('-w', '--load_ckpt', action='store_true', default=False, help="Load last checkpoint.")

    args = parser.parse_args()
    main(**vars(args))


