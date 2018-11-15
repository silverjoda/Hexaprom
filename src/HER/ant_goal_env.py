import gym
import quaternion # Feel free to remove
import numpy as np

class AntG:
    def __init__(self):
        self.env = gym.make("Ant-v3")
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.goal_dim = 3 # x,y,theta (SE2 group)
        self.goal = None
        self.n_episodes = 0
        self.success_rate = 0
        self.xy_dev = 0.1
        self.psi_dev = 0.3


    def reset(self):
        obs = self.env.reset()
        pose = self.get_pose(obs)
        self.goal = self._sample_goal(pose)
        return obs, self.goal


    def _sample_goal(self, pose):
        x, y, psi = pose
        nx = x + np.random.randn() * 1
        ny = y + np.random.randn() * 1
        npsi = y + np.random.randn() * 0.2

        goal = nx, ny, npsi

        if self.reached_goal(pose, goal):
            return self._sample_goal(pose)

        return goal


    def step(self, a):

        # Step inner env
        obs, r, done, info = self.env.step(a)

        # Make relevant pose from observation (x,y,psi)
        pose = self.get_pose(obs)

        # Check if goal has been reached
        reached_goal = self.reached_goal(pose, self.goal)

        # Update success rate
        self._update_stats(reached_goal)

        # Reevaluate termination condition
        done = done or reached_goal

        return obs, r, done, info


    def _update_stats(self, result):
        self.success_rate = (self.success_rate * self.n_episodes + result) / (self.n_episodes + 1)
        self.n_episodes += 1


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