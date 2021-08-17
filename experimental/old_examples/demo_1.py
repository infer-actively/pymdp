import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pymdp.distributions import Categorical, Dirichlet
from pymdp.envs import GridWorldEnv
from pymdp import control
from pymdp import maths

PLOT = False


def plot_beliefs(qs, title=""):
    try:
        values = qs.values[:, 0]
    except AttributeError:
        values = qs[:, 0]
    plt.grid(zorder=0)
    plt.bar(range(qs.shape[0]), values, color="r", zorder=3)
    plt.xticks(range(qs.shape[0]))
    plt.title(title)
    plt.show()


def plot_likelihood(A):
    _ = sns.heatmap(A.values, cmap="OrRd", linewidth=2.5)
    plt.xticks(range(A.shape[1]))
    plt.yticks(range(A.shape[0]))
    plt.title("Likelihood distribution (A)")
    plt.show()


def plot_empirical_prior(B):
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    actions = ["UP", "RIGHT", "DOWN", "LEFT", "STAY"]
    count = 0
    for i in range(3):
        for j in range(2):
            if count >= 5:
                break

            g = sns.heatmap(
                B[:, :, count].values, cmap="OrRd", linewidth=2.5, cbar=False, ax=axes[i, j]
            )
            g.set_title(actions[count])
            count += 1
    fig.delaxes(axes.flatten()[5])
    plt.tight_layout()
    plt.show()


env_shape = [2, 2]
n_states = np.prod(env_shape)

env = GridWorldEnv(shape=env_shape)

likelihood_matrix = env.get_likelihood_dist()
A = Categorical(values=likelihood_matrix)
A.remove_zeros()
if PLOT:
    plot_likelihood(A)

transition_matrix = env.get_transition_dist()
B = Categorical(values=transition_matrix)
B.remove_zeros()
if PLOT:
    plot_empirical_prior(B)

reward_location = 3

C = Categorical(dims=[env.n_states])
C[reward_location] = 1.0
if PLOT:
    plot_beliefs(C, title="Prior preference (C)")

qs = Categorical(dims=[env.n_states])

policy_len = 2
n_control = [env.n_control]
policies = control.construct_policies([n_states], n_control=n_control, policy_len=policy_len)
n_policies = len(policies)
print(f"Total number of policies {n_policies}")
print(policies[0].shape)


def evaluate_policy(policy, qs, A, B, C):
    G = 0
    qs = qs.copy()

    # loop over policy
    policy_len = policy.shape[0]
    for t in range(policy_len):
        # get action
        u = int(policy[t, :])

        # work out expected state
        qs = B[:,:,u].dot(qs)
        # work out expected observations
        qo = A.dot(qs)
        # get entropy
        H = A.entropy()
        # get predicted divergence and uncertainty and novelty
        divergence = maths.kl_divergence(qo, C)
        uncertainty = H.dot(qs)[0, 0]
        G += divergence + uncertainty
    return -G

def infer_action(qs, A, B, C, n_control, policies):
    n_policies = len(policies)

    # negative expected free energy
    neg_G = np.zeros([n_policies, 1])

    for i, policy in enumerate(policies):
        neg_G[i] = evaluate_policy(policy, qs, A, B, C)

    # get distribution over policies
    q_pi = maths.softmax(neg_G)

    # probabilites of control states
    qu = Categorical(dims=n_control)

    # sum probabilites of controls
    for i, policy in enumerate(policies):
        # control state specified by policy
        u = int(policy[0, :])
        # add probability of policy
        qu[u] += q_pi[i]

    # normalize
    qu.normalize()

    # sample control
    u = qu.sample()

    return u

"""
Experiment 
"""

# number of time steps
T = 10

# reset environment
obs = env.reset()
print("Initial Location {}".format(env.state))
# infer initial state
qs = maths.softmax(A[obs, :].log())

# loop over time
for t in range(T):

    # infer action
    action = infer_action(qs, A, B, C, n_control, policies)

    # perform action
    obs = env.step(action)

    # infer new hidden state
    qs = maths.softmax(A[obs, :].log() + B[:,:,action].dot(qs).log())

    # print information
    print("Time step {} Location {}".format(t, env.state))
    if PLOT:
        env.render()
        plot_beliefs(qs, "Beliefs (Qs) at time {}".format(t))

