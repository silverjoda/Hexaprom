import gym

env = gym.make("Snek-v0")
s = env.reset()

while True:
    s,_,_,_=env.step(env.action_space.sample())
    env.render()
