import roboschool, gym;

print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() ]))
exit()
env = gym.make("RoboschoolHumanoidFlagrunHarder-v1")
env.reset()

for i in range(10000):
    env.step(env.action_space.sample())
    env.render()

