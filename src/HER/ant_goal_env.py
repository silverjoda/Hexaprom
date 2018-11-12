import gym
import quaternion # Feel free to remove
import numpy as np

class AntG:
    def __init__(self):
        self.env = gym.make("Antc-v0")
        self.goal = None
        self.n_episodes = 0
        self.successes = 0


    def reset(self):
        obs = self.env.reset()
        pose = self.get_pose(obs)
        goal = self._sample_goal(pose)
        return obs, goal


    def _sample_goal(self, pose):
        x, y, psi = pose
        nx = x + np.random.randn() * 1
        ny = y + np.random.randn() * 1
        npsi = y + np.random.randn() * 0.2
        return nx, ny, npsi


    def step(self, a):
        obs, r, done, info = self.env.step(a)
        pose = self.get_pose(obs)


        return obs, r, done, info


    def get_pose(self, obs):
        x,y = obs[0:2]
        _, _, psi = quaternion.as_euler_angles(np.quaternion(*obs[3:7]))
        return x,y,psi


    def get_success_rate(self):
        return self.successes / self.n_episodes


    def render(self):
        self.env.render()