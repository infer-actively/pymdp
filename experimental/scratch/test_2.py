import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from inferactively import core
from inferactively.distributions import Categorical, Dirichlet
from inferactively.envs import VisualForagingEnv

T = 10

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
A = env.get_likelihood_dist()
B = env.get_transition_dist()

# if you want parameter information gain and/or learning
# pA = Dirichlet(values = A.values * 1e20)
# pB = Dirichlet(values = B.values * 1e20)

C = np.empty(env.n_modalities, dtype=object)
for g, No in enumerate(env.n_observations):
    C[g] = np.zeros(No)

policies, n_control = core.construct_policies(env.n_states, None, 1, [0])

obs = env.reset()

msg = """ === Starting experiment === \n True scene: {} Initial observation {} """
print(msg.format(env.true_scene, obs))
prior = env.get_uniform_posterior()

for t in range(T):

    qs = core.update_posterior_states(A, obs, prior, return_numpy=False)

    msg = """[{}] Inference [location {} / scene {}] 
     Observation [location {} / feature {}] """
    print(msg.format(t, np.argmax(qs[0].values), np.argmax(qs[1].values), obs[0], obs[1]))

    q_pi, efe = core.update_posterior_policies(qs, A, B, C, policies)

    action = core.sample_action(
        q_pi, policies, env.n_control, sampling_type="marginal_action"
    )

    obs = env.step(action)

    msg = """[{}] Action [Saccade to location {}]"""
    print(msg.format(t, action[0]))

    prior = core.get_expected_states(qs, B.log(), action.reshape(1,-1))

