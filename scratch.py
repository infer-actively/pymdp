from gym_minigrid.wrappers import RGBImgPartialObsWrapper
import gym_minigrid
import gym 

env = gym.make('MiniGrid-Empty-5x5-v0')
env = RGBImgPartialObsWrapper(env)
state = env.reset()
print(state['image'].shape)