"""
CORE Implementation:
Implementation of active inference using inferactively. Here, we implement
the necessary computations for hidden state inference and learning (updating parameters
of the generative model) using functions from the core module of inferactively. This manifests as mathematical
manipulations on instances of the Categorical and Dirichlet classes, which represent the various distributions relevant
the posterior and prior beliefs of the agent.
"""

"""
## Imports
"""
import sys
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
## Add `inferactively` module
"""

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from inferactively.envs import GridWorldEnv
from inferactively.distributions import Categorical, Dirichlet
from inferactively import core

"""
## Plotting
# """

def plot_beliefs(belief_dist, title=""):
    values = belief_dist.values[:, 0]
    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), values, color='r', zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title)
    plt.show()
    
def plot_likelihood(A, title=""):
    ax = sns.heatmap(A.values, cmap="OrRd", linewidth=2.5)
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))
    plt.title(title)
    plt.show()

# def plot_empirical_prior(B):
#     fig, axes = plt.subplots(3, 2, figsize=(8, 10))
#     actions = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]
#     count = 0
#     for i in range(3):
#         for j in range(2):
#             if count >= 5:
#                 break

#             g = sns.heatmap(
#                 B[:, :, count].values, cmap="OrRd", linewidth=2.5, cbar=False, ax=axes[i, j]
#             )
#             g.set_title(actions[count])
#             count += 1
#     fig.delaxes(axes.flatten()[5])
#     plt.tight_layout()
#     plt.show()

"""
## Environment

Here, we create a 2x2 `grid_world` environment.

# Likelihoods
We initialise the agent's observation likelihood using the rules provided by the generative process.
@NOTE: Since this is a demo on inference and learning (without actions), we initialise the transition likelihood
of the generative model to be a set of uniform Categorical distributions. Namely, the agent has no strong prior beliefs
about where it is likely to move. However, it will build beliefs about these transitions as it observes itself moving.

@NOTE: Linear indices are in row-major ordering (the default for numpy), to be contrasted with column-major ordering
(the default for MATLAB). This means that the upper-right position in the grid [0,1] corresponds to linear index 1, and the lower-left position [1,0]
corresponds to linear index 2. [0,0] and [1,1] are linear indices 0 and 3, respectively (invariant w.r.t row-major and column-major indexing).

# Posterior (recognition density)
We initialise the posterior beliefs `qs` about hidden states (namely, beliefs about 'where I am') as a flat distribution over the possible states. This requires the
agent to first 'evince' a proprioceptive observation from the environment (e.g. a bodily sensation of where the agent is) before updating its posterior to be centered
on the true, evidence-supported location.

"""

env_shape = [3, 3]
n_states = np.prod(env_shape)

env = GridWorldEnv(shape=env_shape)

likelihood_matrix = env.get_likelihood_dist()
A = Categorical(values=likelihood_matrix)
A.remove_zeros()
plot_likelihood(A,'Observation likelihood')

B = Categorical(values=np.ones((n_states,n_states)))
B.normalize()
plot_likelihood(B,'Transition likelihood')

qs = Categorical(dims=[env.n_states])
qs.normalize()








