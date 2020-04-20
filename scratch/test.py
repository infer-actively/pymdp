import numpy as np

from inferactively import core
from inferactively.distributions import Categorical
from inferactively.envs import VisualForagingEnv

T = 10

env = VisualForagingEnv()
A = env.get_likelihood_dist()
B = env.get_transition_dist()

obs = env.reset()

msg = """ === Starting experiment === \n True scene: {} Initial observation {} """
print(msg.format(env.true_scene, obs))
prior = env.get_uniform_posterior()

for t in range(T):
    action = env.sample_action()
    obs = env.step(action)
    Qs = core.update_posterior_states(A, obs, prior, return_numpy=False)

    msg = """[{}] Inference [location {} / scene {}] 
     Observation [location {} / feature {}] """
    print(msg.format(t, Qs[0].sample(), Qs[1].sample(), obs[0], obs[1]))
