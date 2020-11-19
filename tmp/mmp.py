""" Testing simplest version of MMP 

__date__: 19/11/2020
"""
from typing import Union, Tuple, List

import numpy as np

ObjectArray = np.ndarray
Array = np.ndarray


def obj_array(shape: Union[float, Tuple]) -> ObjectArray:
    return np.empty(shape, dtype=object)


def norm_dist(dist: ObjectArray) -> ObjectArray:
    if len(dist.shape) == 2:
        new_dist = np.zeros_like(dist)
        for c in range(dist.shape[2]):
            new_dist[:, :, c] = np.divide(dist[:, :, c], dist[:, :, c].sum(axis=0))
        return new_dist
    else:
        return np.divide(dist, dist.sum(axis=0))


def onehot(value: int, num_values: int) -> Array:
    arr = np.zeros(num_values)
    arr[value] = 1.0
    return arr


def to_object_arr(arr: Array) -> ObjectArray:
    obj_arr = obj_array(1)
    obj_arr[0] = arr.squeeze()
    return obj_arr


def spm_dot(
    x: ObjectArray, y: Union[ObjectArray, Array], obs_mode: bool = False, omit: List[int] = None
) -> Array:
    if y.dtype == "object":
        dims = (np.arange(0, len(y)) + x.ndim - len(y)).astype(int)
    else:
        if obs_mode:
            dims = np.array([0], dtype=int)
        else:
            dims = np.array([1], dtype=int)
        y = to_object_arr(y)

    if omit is not None:
        dims_omit = np.delete(dims, omit)
        if len(x) == 1:
            y = np.empty([0], dtype=object)
        else:
            y = np.delete(y, dims_omit)

    for d in range(len(y)):
        s = np.ones(np.ndim(x), dtype=int)
        s[dims[d]] = np.shape(y[d])[0]
        x = x * y[d].reshape(tuple(s))
        x = np.sum(x, axis=dims[d], keepdims=True)
    return np.squeeze(x)


def spm_norm(arr: Array) -> Array:
    arr = arr + 1e-16
    normed = np.divide(arr, arr.sum(axis=0))
    return normed


def softmax(arr: Array) -> Array:
    output = arr - arr.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output


def uniform_array(size: int) -> Array:
    return np.ones(size) / size


def mmp(
    A: ObjectArray,
    B: ObjectArray,
    obs_sequence: List[Array],
    _policy: List[Array],
    num_iter: int = 10,
):
    horizon = len(_policy)
    assert horizon == len(obs_sequence)

    num_obs = [modality.shape[0] for modality in obs_sequence[0]]
    num_states = [factor.shape[0] for factor in B]
    num_modalities = len(num_obs)
    num_factors = len(num_states)

    ll = np.ones(tuple(num_states))
    for modality in range(num_modalities):
        ll = ll * spm_dot(A[modality], obs_sequence[0][modality], obs_mode=True)
    ll = np.log(ll + 1e-16)

    qs_seq = [obj_array(num_factors) for t in range(horizon + 1)]
    for t in range(horizon):
        for f in range(num_factors):
            qs_seq[t][f] = uniform_array(num_states[f])

    prior = obj_array(num_factors)
    for f in range(num_factors):
        prior[f] = uniform_array(num_states[f])

    qs_T = obj_array(num_factors)
    for f in range(num_factors):
        qs_T[f] = np.ones(num_states[f])

    policy: Array = np.array(_policy)
    prev_control = np.zeros((1, policy.shape[0]))
    policy = np.vstack((prev_control, policy))

    for _ in range(num_iter):
        for t in range(horizon):
            forward_tensor = obj_array(num_factors)

            for f in range(num_factors):
                
                """ forward """
                if t == 0:
                    log_A = spm_dot(ll, qs_seq[t], omit=[f])
                    forward_msg = np.log(prior[f] + 1e-16)
                else:
                    log_A = np.zeros(num_states[f])
                    _action = int(policy[t - 1, f])
                    _prior = B[f][:, :, _action]
                    _prev_state = qs_seq[t - 1][f]
                    forward_msg = spm_dot(_prior, _prev_state) + 1e-16
                    forward_msg = 0.5 * np.log(forward_msg)
                forward_tensor[f] = 2 * forward_msg

                """ backward """
                if t == horizon - 1:
                    backward_msg = qs_T[f]
                else:
                    _action = int(policy[t + 1, f])
                    _prior = spm_norm(B[f][:, :, _action].T)
                    _next_state = qs_seq[t + 1][f]
                    backward_msg = spm_dot(_prior, _next_state) + 1e-16
                    backward_msg = 0.5 * np.log(backward_msg)

                qs_seq[t][f] = softmax(log_A + forward_msg + backward_msg)

    return qs_seq


if __name__ == "__main__":

    num_obs = [2, 2]
    num_states = [2, 2]
    num_controls = [3, 3]
    num_modalities = len(num_obs)
    num_factors = len(num_states)

    A = obj_array(num_modalities)
    for modality, modality_obs in enumerate(num_obs):
        modality_shape = [modality_obs] + num_states
        modality_dist = np.random.rand(*modality_shape)
        A[modality] = norm_dist(modality_dist)

    assert len(num_controls) == len(num_states)
    B = obj_array(num_factors)
    for factor in range(num_factors):
        factor_shape = (num_states[factor], num_states[factor], num_controls[factor])
        factor_dist = np.random.rand(*factor_shape)
        B[factor] = norm_dist(factor_dist)

    obs = obj_array(num_modalities)
    for modality in range(num_modalities):
        obs[modality] = onehot(0, num_obs[modality])

    obs_sequence = [obs, obs]
    control = np.array([0, 0])
    policy = [control, control]

    mmp(A, B, obs_sequence, policy)