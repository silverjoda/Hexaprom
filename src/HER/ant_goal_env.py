import gym
import quaternion # Feel free to remove
import numpy as np
from collections import deque

class AntG:
    def __init__(self):
        self.env = gym.make("Ant-v3")
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.goal_dim = 3 # x,y,theta (SE2 group)
        self.goal = None
        self.n_episodes = 0
        self.success_rate = 0
        self.prev_success = False
        self.success_queue = deque(maxlen=300)
        self.xy_dev = 0.1
        self.psi_dev = 0.3

    # TODO: DEBUG ENVIRONMENT

    def reset(self):
        obs = self.env.reset()
        self.goal = self._sample_goal(self.get_pose(obs))
        return obs, self.goal


    def _sample_goal(self, pose):
        while True:
            x, y, psi = pose
            nx = x + np.random.randn() * (2. + 3 * self.success_rate)
            ny = y + np.random.randn() * (2. + 3 * self.success_rate)
            npsi = y + np.random.randn() * (0.3 + 1 * self.success_rate)

            goal = nx, ny, npsi

            if not self.reached_goal(pose, goal):
                break

        return goal


    def step(self, a):

        # Step inner env
        obs, _, done, info = self.env.step(a)

        # Make relevant pose from observation (x,y,psi)
        pose = self.get_pose(obs)

        # Check if goal has been reached
        reached_goal = self.reached_goal(pose, self.goal)

        # Reevaluate termination condition
        done = done or reached_goal

        if reached_goal:
            print("SUCCESS")

        # Update success rate
        if done:
            self._update_stats(reached_goal)

        r = 1. if reached_goal else 0.

        return obs, r, done, info


    def _update_stats(self, reached_goal):
        self.success_queue.append(1. if reached_goal else 0.)
        self.success_rate = np.mean(self.success_queue)


    def reached_goal(self, pose, goal):
        x,y,psi = pose
        xg,yg,psig = goal
        return (x-xg)**2 < self.xy_dev and (y-yg)**2 < self.xy_dev and (psi-psig)**2 < self.psi_dev


    def get_pose(self, obs):
        x,y = obs[0:2]
        _, _, psi = quaternion.as_euler_angles(np.quaternion(*obs[3:7]))
        return x,y,psi


    def render(self):
        self.env.render()