import roboschool, gym;

print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() ]))

env = gym.make("Centipede30-v0")
env.reset()

print(env.action_space, env.observation_space)

for i in range(10000):
    obs,_,_,_ = env.step(env.action_space.sample())
    env.render()

