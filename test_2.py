import numpy as np

from inferactively import core
from inferactively.distributions import Categorical, Dirichlet
from inferactively.envs import VisualForagingEnv

T = 10

print("""Initializing scene configuration with two scenes that share one feature
 in common. Therefore each scene has only one disambiguating feature\n""")

scenes = np.zeros( (2,2,2) )
scenes[0][0,0] = 1
scenes[0][1,1] = 2
scenes[1][1,1] = 2
scenes[1][1,0] = 3

env = VisualForagingEnv(scenes = scenes,n_features = 3)
A = env.get_likelihood_dist()
B = env.get_transition_dist()

# if you want parameter information gain and/or learning
# pA = Dirichlet(values = A.values * 1e20)
# pB = Dirichlet(values = B.values * 1e20)

C = np.empty(env.n_modalities,dtype=object)
for g, No in enumerate(env.n_observations):
    C[g] = np.zeros(No)

_,possible_policies = core.constructNu(env.n_states, env.n_factors, [0], 1)

obs = env.reset()

msg = """ === Starting experiment === \n True scene: {} Initial observation {} """
print(msg.format(env.true_scene, obs))
prior = env.get_uniform_posterior()

for t in range(T):
    
    Qs = core.update_posterior_states(A, obs, prior, return_numpy=False)

    msg = """[{}] Inference [location {} / scene {}] 
     Observation [location {} / feature {}] """
    print(msg.format(t, Qs[0].sample(), Qs[1].sample(), obs[0], obs[1]))

    Q_pi, _ = core.update_posterior_policies(Qs, A, B, C, possible_policies)

    action = core.sample_action(Q_pi, possible_policies, env.n_control, sampling_type="marginal_action")

    obs = env.step(action)

    msg = """[{}] Action [Saccade to location {}]"""
    print(msg.format(t,action[0]))

    prior = core.get_expected_states(Qs, B, action)

