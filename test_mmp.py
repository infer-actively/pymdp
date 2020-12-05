import numpy as np

from inferactively.core.utils import random_A_matrix, random_B_matrix, obj_array, onehot
from inferactively.core.maths import get_joint_likelihood_seq
from inferactively.core.algos import run_mmp_v2


def rand_onehot_obs(num_obs):
    if type(num_obs) is int:
        num_obs = [num_obs]
    obs = obj_array(len(num_obs))
    for i in range(len(num_obs)):
        ob = np.random.randint(num_obs[i])
        obs[i] = onehot(ob, num_obs[i])
    return obs


def rand_controls(num_controls):
    if type(num_controls) is int:
        num_controls = [num_controls]
    controls = np.zeros(len(num_controls))
    for i in range(len(num_controls)):
        controls[i] = np.random.randint(num_controls[i])
    return controls


if __name__ == "__main__":
    past_len = 4
    future_len = 4
    num_states = 2
    num_controls = 3
    num_obs = 5

    A = random_A_matrix(num_obs, num_states)
    B = random_B_matrix(num_states, num_controls)
    prev_obs = [rand_onehot_obs(num_obs) for _ in range(past_len)]
    prev_actions = np.array([rand_controls(num_controls) for _ in range(past_len)])
    policy = np.array([rand_controls(num_controls) for _ in range(future_len)])
                                                     
    ll_seq = get_joint_likelihood_seq(A, prev_obs, num_states)
    qs_seq = run_mmp_v2(A, B, ll_seq, policy, grad_descent=True)
    for t, qs in enumerate(qs_seq):
        print(f"Step {t} shape {[el.shape for el in qs]}")
