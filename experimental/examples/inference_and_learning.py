"""
CORE Implementation:

Implementation of active inference using inferactively. Here, we implement
the necessary computations for hidden state inference and learning (updating parameters
of the generative model) using functions from the `core` module of inferactively. 

This ends up being simple mathematical manipulations of instances of the Categorical and Dirichlet classes, 
which represent the various distributions encoding posterior and prior beliefs of the agent.
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
## Plotting functions (come in handy later on)
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

"""
## Environment

We initialize a 3 x 3 `grid_world` environment and create a sequence of hidden states.
The transition dynamics here are not explicitly defined - rather, we sample a random sequence of
states and repeat it 3 times, allowing the agent to build a model of the transition dynamics via
frequentist-style updates to the Dirichlet parameters `b` (see below "Generative model" section).

"""

env_shape = [3, 3]
n_states = np.prod(env_shape)

env = GridWorldEnv(shape=env_shape)

gp_likelihood = env.get_likelihood_dist()

# create a random sequence of 5 states
state_sequence = [np.random.randint(n_states) for i in range(4)]

# three repetitions
s = state_sequence * 3 

T = len(s) - 1 # duration of the environmental dynamics (minus 1 because we do the first observation/state ourselves)


"""
## Generative model

Now we initialise the generative model of the agent

# Likelihoods
We initialise the agent's observation likelihood using the rules provided by the generative process.
@NOTE: Since this is a demo on inference and learning (without actions), we initialise the transition likelihood
of the generative model to be a set of uniform Categorical distributions. Namely, the agent has no strong prior beliefs
about where it is likely to move. However, it will build beliefs about these transitions as it observes itself moving.

@NOTE: Linear indices are in row-major ordering (the default for numpy), to be contrasted with column-major ordering
(the default for MATLAB). This means that the upper-right position in the grid [0,2] corresponds to linear index 2, and the lower-left position [2,0]
corresponds to linear index 6. [0,0] and [2,2] are linear indices 0 and 8, respectively (invariant w.r.t row-major and column-major indexing).

# Prior
We initialise the agent's prior over hidden states at the start to be a flat distribution (i.e. the agent has no strong beliefs about what state it is starting in)

# Posterior (recognition density)
We initialise the posterior beliefs `qs` about hidden states (namely, beliefs about 'where I am') as a flat distribution over the possible states. This requires the
agent to first gather a proprioceptive observation from the environment (e.g. a bodily sensation of where it feels itself to be) before updating its posterior to be centered
on the true, evidence-supported location.
"""

likelihood_matrix = env.get_likelihood_dist()
A = Categorical(values=likelihood_matrix)
A.remove_zeros()
plot_likelihood(A,'Observation likelihood')

b = Dirichlet(values = np.ones((n_states,n_states)))      
B = b.mean()
plot_likelihood(B,'Initial transition likelihood')

D = Categorical(values = np.ones(n_states)) 
D.normalize()

qs = Categorical(dims=[env.n_states])
qs.normalize()


"""
Run the dynamics of the environment and inference. Start by eliciting one observation from the environment
"""


# reset environment
first_state = env.reset(init_state = s[0] )

# get an observation, given the state
first_obs = gp_likelihood[:,first_state]

# turn observation into an index
first_obs = np.where(first_obs)[0][0]

print("Initial Location {}".format(first_state))
print("Initial Observation {}".format(first_obs))

# infer initial state, given first observation
qs = core.softmax(A[first_obs, :].log() + D.log(), return_numpy=False)

# loop over time
for t in range(T):

    qs_past = qs.copy()

    s_t = env.set_state(s[t+1])

    # evoke observation, given the state
    obs = gp_likelihood[:,s_t]

    # turn observation into an index
    obs = np.where(obs)[0][0]

    # get transition likelihood
    B = b.mean()

    # infer new hidden state
    qs = core.softmax(A[obs, :].log() + B.dot(qs_past).log(), return_numpy=False)

    # update beliefs about the transition likelihood (i.e. 'learning')

    b += qs.cross(qs_past,return_numpy=True) # update transition likelihood

    # print information
    print("Time step {} Location {}".format(t, env.state))

    # env.render()
    plot_beliefs(qs, "Beliefs (Qs) at time {}".format(t))

B = b.mean()
plot_likelihood(B,'Final transition likelihood')













