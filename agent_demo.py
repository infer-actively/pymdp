import numpy as np

from inferactively import core
from inferactively.distributions import Categorical, Dirichlet
from inferactively.agent import Agent
from inferactively.envs import VisualForagingEnv

print(
    """Initializing scene configuration with two scenes that share one feature
 in common. Therefore each scene has only one disambiguating feature\n"""
)

scenes = np.zeros((2, 2, 2))
scenes[0][0, 0] = 1
scenes[0][1, 1] = 2
scenes[1][1, 1] = 2
scenes[1][1, 0] = 3

env = VisualForagingEnv(scenes=scenes, n_features=3)

agent = Agent(A=env.get_likelihood_dist(), B=env.get_transition_dist(), control_fac_idx=[0])


T = 10

obs = env.reset()

for t in range(T):

    qx = agent.infer_states(obs)

    q_pi, efe = agent.infer_policies()

    action = agent.sample_action()

    msg = """[Step {}] Action: [Saccade to location {}]"""
    print(msg.format(t, action[0]))

    obs = env.step(action)

    msg = """[Step {}] Observation: [Location {}, Feature {}]"""
    print(msg.format(t, obs[0], obs[1]))

