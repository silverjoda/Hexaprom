



import torch
from sys import exit

x = torch.randn(1, 5, requires_grad=True)
xf = torch.rfft(x, 2)
print(xf)
sum = torch.sum(xf)
sum.backward()
print(x.grad)


exit()
import roboschool, gym;

print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() ]))

env = gym.make("RoboschoolHumanoid-v1")
env.reset()

print(env.action_space, env.observation_space)


for i in range(10000):
    env.step(env.action_space.sample())
    env.render()

