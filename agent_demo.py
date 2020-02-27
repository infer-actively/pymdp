# %% Imports

import numpy as np

from inferactively import core
from inferactively.distributions import Categorical, Dirichlet
from inferactively.agent import Agent
from inferactively.envs import VisualForagingEnv

# %% Initialize an agent, get an observation, and perform inference about states

print("""Initializing scene configuration with two scenes that share one feature
 in common. Therefore each scene has only one disambiguating feature\n""")

scenes = np.zeros( (2,2,2) )
scenes[0][0,0] = 1
scenes[0][1,1] = 2
scenes[1][1,1] = 2
scenes[1][1,0] = 3

genProcess = VisualForagingEnv(scenes = scenes,n_features = 3)

genModel = Agent(A=genProcess.get_likelihood_dist(), B = genProcess.get_transition_dist(), control_fac_idx = [0])

# %% Main loop

T = 10

obs = genProcess.reset()

for t in range(T):
    
    qx = genModel.infer_states(obs)

    q_pi, efe = genModel.infer_policies()

    action = genModel.sample_action()

    msg = """[Step {}] Action: [Saccade to location {}]"""
    print(msg.format(t,action[0]))

    obs = genProcess.step(action)

    msg = """[Step {}] Observation: [Location {}, Feature {}]"""
    print(msg.format(t,obs[0],obs[1]))

