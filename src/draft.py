import roboschool, gym;
import torch, time


print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() ]))

env = gym.make("Centipede8-v0")
env.reset()

print(env.action_space, env.observation_space)

for i in range(10000):
    #obs,_,_,_ = env.step(env.action_space.sample())
    obs, _, _, _ = env.step([-1,-1,1,-1] + [-1,-1,1,-1] + [+1,+1] + [0]*12)
    print(obs)
    env.render()

