#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Utility functions

__author__: Conor Heins, Alexander Tschantz, Brennan Klein
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
import itertools

EPS_VAL = 1e-16 # global constant for use in norm_dist()

def sample(probabilities):
    probabilities = probabilities.squeeze() if len(probabilities) > 1 else probabilities
    sample_onehot = np.random.multinomial(1, probabilities)
    return np.where(sample_onehot == 1)[0][0]

def sample_obj_array(arr):
    """ 
    Sample from set of Categorical distributions, stored in the sub-arrays of an object array 
    """
    
    samples = [sample(arr_i) for arr_i in arr]

    return samples

def obj_array(num_arr):
    """
    Creates a generic object array with the desired number of sub-arrays, given by `num_arr`
    """
    return np.empty(num_arr, dtype=object)

def obj_array_zeros(shape_list):
    """ 
    Creates a numpy object array whose sub-arrays are 1-D vectors
    filled with zeros, with shapes given by shape_list[i]
    """
    arr = obj_array(len(shape_list))
    for i, shape in enumerate(shape_list):
        arr[i] = np.zeros(shape)
    return arr

def initialize_empty_A(num_obs, num_states):
    """ 
    Initializes an empty observation likelihood array or `A` array using a list of observation-modality dimensions (`num_obs`)
    and hidden state factor dimensions (`num_states`)
    """

    A_shape_list = [ [no] + num_states for no in num_obs]
    return obj_array_zeros(A_shape_list)

def initialize_empty_B(num_states, num_controls):
    """ 
    Initializes an empty (controllable) transition likelihood array or `B` array using a list of hidden state factor dimensions (`num_states`)
    and control factor dimensions (`num_controls)
    """

    B_shape_list = [ [ns, ns, num_controls[f]] for f, ns in enumerate(num_states)]
    return obj_array_zeros(B_shape_list)

def obj_array_uniform(shape_list):
    """ 
    Creates a numpy object array whose sub-arrays are uniform Categorical
    distributions with shapes given by shape_list[i]. The shapes (elements of shape_list)
    can either be tuples or lists.
    """
    arr = obj_array(len(shape_list))
    for i, shape in enumerate(shape_list):
        arr[i] = norm_dist(np.ones(shape))
    return arr

def obj_array_ones(shape_list, scale = 1.0):
    arr = obj_array(len(shape_list))
    for i, shape in enumerate(shape_list):
        arr[i] = scale * np.ones(shape)
    
    return arr

def onehot(value, num_values):
    arr = np.zeros(num_values)
    arr[value] = 1.0
    return arr

def random_A_matrix(num_obs, num_states):
    if type(num_obs) is int:
        num_obs = [num_obs]
    if type(num_states) is int:
        num_states = [num_states]
    num_modalities = len(num_obs)

    A = obj_array(num_modalities)
    for modality, modality_obs in enumerate(num_obs):
        modality_shape = [modality_obs] + num_states
        modality_dist = np.random.rand(*modality_shape)
        A[modality] = norm_dist(modality_dist)
    return A

def random_B_matrix(num_states, num_controls):
    if type(num_states) is int:
        num_states = [num_states]
    if type(num_controls) is int:
        num_controls = [num_controls]
    num_factors = len(num_states)
    assert len(num_controls) == len(num_states)

    B = obj_array(num_factors)
    for factor in range(num_factors):
        factor_shape = (num_states[factor], num_states[factor], num_controls[factor])
        factor_dist = np.random.rand(*factor_shape)
        B[factor] = norm_dist(factor_dist)
    return B

def random_single_categorical(shape_list):
    """
    Creates a random 1-D categorical distribution (or set of 1-D categoricals, e.g. multiple marginals of different factors) and returns them in an object array 
    """
    
    num_sub_arrays = len(shape_list)

    out = obj_array(num_sub_arrays)

    for arr_idx, shape_i  in enumerate(shape_list):
        out[arr_idx] = norm_dist(np.random.rand(shape_i))
    
    return out

def construct_controllable_B(num_states, num_controls):
    """
    Generates a fully controllable transition likelihood array, where each 
    action (control state) corresponds to a move to the n-th state from any 
    other state, for each control factor
    """

    num_factors = len(num_states)

    B = obj_array(num_factors)
    for factor, c_dim in enumerate(num_controls):
        tmp = np.eye(c_dim)[:, :, np.newaxis]
        tmp = np.tile(tmp, (1, 1, c_dim))
        B[factor] = tmp.transpose(1, 2, 0)

    return B

def dirichlet_like(template_categorical, scale = 1.0):
    """
    Helper function to construct a Dirichlet distribution based on an existing Categorical distribution
    """ 

    if not is_obj_array(template_categorical):
        warnings.warn(
                    "Input array is not an object array...\
                    Casting the input to an object array"
                )
        template_categorical = to_obj_array(template_categorical)

    n_sub_arrays = len(template_categorical)

    dirichlet_out = obj_array(n_sub_arrays)
    
    for i, arr in enumerate(template_categorical):
        dirichlet_out[i] = scale * arr

    return dirichlet_out

def get_model_dimensions(A=None, B=None):

    if A is None and B is None:
        raise ValueError(
                    "Must provide either `A` or `B`"
                )

    if A is not None:
        num_obs = [a.shape[0] for a in A] if is_obj_array(A) else [A.shape[0]]
        num_modalities = len(num_obs)
    else:
        num_obs, num_modalities = None, None
    
    if B is not None:
        num_states = [b.shape[0] for b in B] if is_obj_array(B) else [B.shape[0]]
        num_factors = len(num_states)
    else:
        if A is not None:
            num_states = list(A[0].shape[1:]) if is_obj_array(A) else list(A.shape[1:])
            num_factors = len(num_states)
        else:
            num_states, num_factors = None, None
    
    return num_obs, num_states, num_modalities, num_factors

def get_model_dimensions_from_labels(model_labels):

    modalities = model_labels['observations']
    num_modalities = len(modalities.keys())
    num_obs = [len(modalities[modality]) for modality in modalities.keys()]

    factors = model_labels['states']
    num_factors = len(factors.keys())
    num_states = [len(factors[factor]) for factor in factors.keys()]

    if 'actions' in model_labels.keys():

        controls = model_labels['actions']
        num_control_fac = len(controls.keys())
        num_controls = [len(controls[cfac]) for cfac in controls.keys()]

        return num_obs, num_modalities, num_states, num_factors, num_controls, num_control_fac
    else:
        return num_obs, num_modalities, num_states, num_factors


def norm_dist(dist):
    """ Normalizes a Categorical probability distribution (or set of them) assuming sufficient statistics are stored in leading dimension"""
    if dist.ndim == 3:
        new_dist = np.zeros_like(dist)
        for c in range(dist.shape[2]):
            new_dist[:, :, c] = np.divide(dist[:, :, c], dist[:, :, c].sum(axis=0))
        return new_dist
    else:
        return np.divide(dist, dist.sum(axis=0))

def norm_dist_obj_arr(obj_arr):

    normed_obj_array = obj_array(len(obj_arr))
    for i, arr in enumerate(obj_arr):
        normed_obj_array[i] = norm_dist(arr)
    
    return normed_obj_array

def is_normalized(dist):
    """
    Utility function for checking whether a single distribution or set of conditional categorical distributions is normalized.
    Returns True if all distributions integrate to 1.0
    """

    if is_obj_array(dist):
        normed_arrays = []
        for i, arr in enumerate(dist):
            column_sums = arr.sum(axis=0)
            normed_arrays.append(np.allclose(column_sums, np.ones_like(column_sums)))
        out = all(normed_arrays)
    else:
        column_sums = dist.sum(axis=0)
        out = np.allclose(column_sums, np.ones_like(column_sums))
    
    return out

def is_obj_array(arr):
    return arr.dtype == "object"

def to_obj_array(arr):
    if is_obj_array(arr):
        return arr
    obj_array_out = obj_array(1)
    obj_array_out[0] = arr.squeeze()
    return obj_array_out

def obj_array_from_list(list_input):
    """
    Takes a list of `numpy.ndarray` and converts them to a `numpy.ndarray` of `dtype = object`
    """
    return np.array(list_input, dtype = object)

def process_observation_seq(obs_seq, n_modalities, n_observations):
    """
    Helper function for formatting observations    

        Observations can either be `int` (converted to one-hot)
        or `tuple` (obs for each modality), or `list` (obs for each modality)
        If list, the entries could be object arrays of one-hots, in which
        case this function returns `obs_seq` as is.
    """
    proc_obs_seq = obj_array(len(obs_seq))
    for t, obs_t in enumerate(obs_seq):
        proc_obs_seq[t] = process_observation(obs_t, n_modalities, n_observations)
    return proc_obs_seq

def process_observation(obs, num_modalities, num_observations):
    """
    Helper function for formatting observations    
    USAGE NOTES:
    - If `obs` is a 1D numpy array, it must be a one-hot vector, where one entry (the entry of the observation) is 1.0 
    and all other entries are 0. This therefore assumes it's a single modality observation. If these conditions are met, then
    this function will return `obs` unchanged. Otherwise, it'll throw an error.
    - If `obs` is an int, it assumes this is a single modality observation, whose observation index is given by the value of `obs`. This function will convert
    it to be a one hot vector.
    - If `obs` is a list, it assumes this is a multiple modality observation, whose len is equal to the number of observation modalities,
    and where each entry `obs[m]` is the index of the observation, for that modality. This function will convert it into an object array
    of one-hot vectors.
    - If `obs` is a tuple, same logic as applies for list (see above).
    - if `obs` is a numpy object array (array of arrays), this function will return `obs` unchanged.
    """

    if isinstance(obs, np.ndarray) and not is_obj_array(obs):
        assert num_modalities == 1, "If `obs` is a 1D numpy array, `num_modalities` must be equal to 1"
        assert len(np.where(obs)[0]) == 1, "If `obs` is a 1D numpy array, it must be a one hot vector (e.g. np.array([0.0, 1.0, 0.0, ....]))"

    if isinstance(obs, (int, np.integer)):
        obs = onehot(obs, num_observations[0])

    if isinstance(obs, tuple) or isinstance(obs,list):
        obs_arr_arr = obj_array(num_modalities)
        for m in range(num_modalities):
            obs_arr_arr[m] = onehot(obs[m], num_observations[m])
        obs = obs_arr_arr

    return obs

def convert_observation_array(obs, num_obs):
    """
    Converts from SPM-style observation array to infer-actively one-hot object arrays.
    
    Parameters
    ----------
    - 'obs' [numpy 2-D nd.array]:
        SPM-style observation arrays are of shape (num_modalities, T), where each row 
        contains observation indices for a different modality, and columns indicate 
        different timepoints. Entries store the indices of the discrete observations 
        within each modality. 

    - 'num_obs' [list]:
        List of the dimensionalities of the observation modalities. `num_modalities` 
        is calculated as `len(num_obs)` in the function to determine whether we're 
        dealing with a single- or multi-modality 
        case.

    Returns
    ----------
    - `obs_t`[list]: 
        A list with length equal to T, where each entry of the list is either a) an object 
        array (in the case of multiple modalities) where each sub-array is a one-hot vector 
        with the observation for the correspond modality, or b) a 1D numpy array (in the case
        of one modality) that is a single one-hot vector encoding the observation for the 
        single modality.
    """

    T = obs.shape[1]
    num_modalities = len(num_obs)

    # Initialise the output
    obs_t = []
    # Case of one modality
    if num_modalities == 1:
        for t in range(T):
            obs_t.append(onehot(obs[0, t] - 1, num_obs[0]))
    else:
        for t in range(T):
            obs_AoA = obj_array(num_modalities)
            for g in range(num_modalities):
                # Subtract obs[g,t] by 1 to account for MATLAB vs. Python indexing
                # (MATLAB is 1-indexed)
                obs_AoA[g] = onehot(obs[g, t] - 1, num_obs[g])
            obs_t.append(obs_AoA)

    return obs_t

def insert_multiple(s, indices, items):
    for idx in range(len(items)):
        s.insert(indices[idx], items[idx])
    return s

def reduce_a_matrix(A):
    """
    Utility function for throwing away dimensions (lagging dimensions, hidden state factors)
    of a particular A matrix that are independent of the observation. 
    Parameters:
    ==========
    - `A` [np.ndarray]:
        The A matrix or likelihood array that encodes probabilistic relationship
        of the generative model between hidden state factors (lagging dimensions, columns, slices, etc...)
        and observations (leading dimension, rows). 
    Returns:
    =========
    - `A_reduced` [np.ndarray]:
        The reduced A matrix, missing the lagging dimensions that correspond to hidden state factors
        that are statistically independent of observations
    - `original_factor_idx` [list]:
        List of the indices (in terms of the original dimensionality) of the hidden state factors
        that are maintained in the A matrix (and thus have an informative / non-degenerate relationship to observations
    """

    o_dim, num_states = A.shape[0], A.shape[1:]
    idx_vec_s = [slice(0, o_dim)]  + [slice(ns) for _, ns in enumerate(num_states)]

    original_factor_idx = []
    excluded_factor_idx = [] # the indices of the hidden state factors that are independent of the observation and thus marginalized away
    for factor_i, ns in enumerate(num_states):

        level_counter = 0
        break_flag = False
        while level_counter < ns and break_flag is False:
            idx_vec_i = idx_vec_s.copy()
            idx_vec_i[factor_i+1] = slice(level_counter,level_counter+1,None)
            if not np.isclose(A.mean(axis=factor_i+1), A[tuple(idx_vec_i)].squeeze()).all():
                break_flag = True # this means they're not independent
                original_factor_idx.append(factor_i)
            else:
                level_counter += 1
        
        if break_flag is False:
            excluded_factor_idx.append(factor_i+1)
    
    A_reduced = A.mean(axis=tuple(excluded_factor_idx)).squeeze()

    return A_reduced, original_factor_idx

def construct_full_a(A_reduced, original_factor_idx, num_states):
    """
    Utility function for reconstruction a full A matrix from a reduced A matrix, using known factor indices
    to tile out the reduced A matrix along the 'non-informative' dimensions
    Parameters:
    ==========
    - `A_reduced` [np.ndarray]:
        The reduced A matrix or likelihood array that encodes probabilistic relationship
        of the generative model between hidden state factors (lagging dimensions, columns, slices, etc...)
        and observations (leading dimension, rows). 
    - `original_factor_idx` [list]:
        List of hidden state indices in terms of the full hidden state factor list, that comprise
        the lagging dimensions of `A_reduced`
    - `num_states` [list]:
        The list of all the dimensionalities of hidden state factors in the full generative model.
        `A_reduced.shape[1:]` should be equal to `num_states[original_factor_idx]`
    Returns:
    =========
    - `A` [np.ndarray]:
        The full A matrix, containing all the lagging dimensions that correspond to hidden state factors, including
        those that are statistically independent of observations
    
    @ NOTE: This is the "inverse" of the reduce_a_matrix function, 
    i.e. `reduce_a_matrix(construct_full_a(A_reduced, original_factor_idx, num_states)) == A_reduced, original_factor_idx`
    """

    o_dim = A_reduced.shape[0] # dimensionality of the support of the likelihood distribution (i.e. the number of observation levels)
    full_dimensionality = [o_dim] + num_states # full dimensionality of the output (`A`)
    fill_indices = [0] +  [f+1 for f in original_factor_idx] # these are the indices of the dimensions we need to fill for this modality
    fill_dimensions = np.delete(full_dimensionality, fill_indices) 

    original_factor_dims = [num_states[f] for f in original_factor_idx] # dimensionalities of the relevant factors
    prefilled_slices = [slice(0, o_dim)] + [slice(0, ns) for ns in original_factor_dims] # these are the slices that are filled out by the provided `A_reduced`

    A = np.zeros(full_dimensionality)

    for item in itertools.product(*[list(range(d)) for d in fill_dimensions]):
        slice_ = list(item)
        A_indices = insert_multiple(slice_, fill_indices, prefilled_slices) #here we insert the correct values for the fill indices for this slice                    
        A[tuple(A_indices)] = A_reduced
    
    return A

def create_A_matrix_stub(model_labels):

    num_obs, _, num_states, _= get_model_dimensions_from_labels(model_labels)

    obs_labels, state_labels = model_labels['observations'], model_labels['states']

    state_combinations = pd.MultiIndex.from_product(list(state_labels.values()), names=list(state_labels.keys()))
    num_state_combos = np.prod(num_states)
    # num_rows = (np.array(num_obs) * num_state_combos).sum()
    num_rows = sum(num_obs)

    cell_values = np.zeros((num_rows, len(state_combinations)))

    obs_combinations = []
    for modality in obs_labels.keys():
        levels_to_combine = [[modality]] + [obs_labels[modality]]
        # obs_combinations += num_state_combos * list(itertools.product(*levels_to_combine))
        obs_combinations += list(itertools.product(*levels_to_combine))


    obs_combinations = pd.MultiIndex.from_tuples(obs_combinations, names = ["Modality", "Level"])

    A_matrix = pd.DataFrame(cell_values, index = obs_combinations, columns=state_combinations)

    return A_matrix

def create_B_matrix_stubs(model_labels):

    _, _, num_states, _, num_controls, _ = get_model_dimensions_from_labels(model_labels)

    state_labels = model_labels['states']
    action_labels = model_labels['actions']

    B_matrices = {}

    for f_idx, factor in enumerate(state_labels.keys()):

        control_fac_name = list(action_labels)[f_idx]
        factor_list = [state_labels[factor]] + [action_labels[control_fac_name]]

        prev_state_action_combos = pd.MultiIndex.from_product(factor_list, names=[factor, list(action_labels.keys())[f_idx]])

        num_state_action_combos = num_states[f_idx] * num_controls[f_idx]

        num_rows = num_states[f_idx]

        cell_values = np.zeros((num_rows, num_state_action_combos))

        next_state_list = state_labels[factor]
        
        B_matrix_f = pd.DataFrame(cell_values, index = next_state_list, columns=prev_state_action_combos)

        B_matrices[factor] = B_matrix_f

    return B_matrices

def read_A_matrix(path, num_hidden_state_factors):
    raw_table = pd.read_excel(path, header=None)
    level_counts = {
        "index": raw_table.iloc[0, :].dropna().index[0] + 1,
        "header": raw_table.iloc[0, :].dropna().index[0] + num_hidden_state_factors - 1,
    }
    return pd.read_excel(
        path,
        index_col=list(range(level_counts["index"])),
        header=list(range(level_counts["header"]))
        ).astype(np.float64)

def read_B_matrices(path):

    all_sheets = pd.read_excel(path, sheet_name = None, header=None)

    level_counts = {}
    for sheet_name, raw_table in all_sheets.items():
    
        level_counts[sheet_name] = {
            "index": raw_table.iloc[0, :].dropna().index[0]+1,
            "header": raw_table.iloc[0, :].dropna().index[0]+2,
        }

    stub_dict = {}
    for sheet_name, level_counts_sheet in level_counts.items():
        sheet_f = pd.read_excel(
            path,
            sheet_name = sheet_name,
            index_col=list(range(level_counts_sheet["index"])),
            header=list(range(level_counts_sheet["header"]))
            ).astype(np.float64)
        stub_dict[sheet_name] = sheet_f
        
    return stub_dict

def convert_A_stub_to_ndarray(A_stub, model_labels):
    """
    This function converts a multi-index pandas dataframe `A_stub` into an object array of different
    A matrices, one per observation modality. 
    """

    num_obs, num_modalities, num_states, num_factors = get_model_dimensions_from_labels(model_labels)

    A = obj_array(num_modalities)

    for g, modality_name in enumerate(model_labels['observations'].keys()):
        A[g] = A_stub.loc[modality_name].to_numpy().reshape(num_obs[g], *num_states)
        assert (A[g].sum(axis=0) == 1.0).all(), 'A matrix not normalized! Check your initialization....\n'

    return A

def convert_B_stubs_to_ndarray(B_stubs, model_labels):
    """
    This function converts a list of multi-index pandas dataframes `B_stubs` into an object array
    of different B matrices, one per hidden state factor
    """

    _, _, num_states, num_factors, num_controls, num_control_fac  = get_model_dimensions_from_labels(model_labels)

    B = obj_array(num_factors)

    for f, factor_name in enumerate(B_stubs.keys()):
        
        B[f] = B_stubs[factor_name].to_numpy().reshape(num_states[f], num_states[f], num_controls[f])
        assert (B[f].sum(axis=0) == 1.0).all(), 'B matrix not normalized! Check your initialization....\n'

    return B

# def build_belief_array(qx):

#     """
#     This function constructs array-ified (not nested) versions
#     of the posterior belief arrays, that are separated 
#     by policy, timepoint, and hidden state factor
#     """

#     num_policies = len(qx)
#     num_timesteps = len(qx[0])
#     num_factors = len(qx[0][0])

#     if num_factors > 1:
#         belief_array = obj_array(num_factors)
#         for factor in range(num_factors):
#             belief_array[factor] = np.zeros( (num_policies, qx[0][0][factor].shape[0], num_timesteps) )
#         for policy_i in range(num_policies):
#             for timestep in range(num_timesteps):
#                 for factor in range(num_factors):
#                     belief_array[factor][policy_i, :, timestep] = qx[policy_i][timestep][factor]
#     else:
#         num_states = qx[0][0][0].shape[0]
#         belief_array = np.zeros( (num_policies, num_states, num_timesteps) )
#         for policy_i in range(num_policies):
#             for timestep in range(num_timesteps):
#                 belief_array[policy_i, :, timestep] = qx[policy_i][timestep][0]
    
#     return belief_array

def build_xn_vn_array(xn):

    """
    This function constructs array-ified (not nested) versions
    of the posterior xn (beliefs) or vn (prediction error) arrays, that are separated 
    by iteration, hidden state factor, timepoint, and policy
    """

    num_policies = len(xn)
    num_itr = len(xn[0])
    num_factors = len(xn[0][0])

    if num_factors > 1:
        xn_array = obj_array(num_factors)
        for factor in range(num_factors):
            num_states, infer_len = xn[0][0][factor].shape
            xn_array[factor] = np.zeros( (num_itr, num_states, infer_len, num_policies) )
        for policy_i in range(num_policies):
            for itr in range(num_itr):
                for factor in range(num_factors):
                    xn_array[factor][itr,:,:,policy_i] = xn[policy_i][itr][factor]
    else:
        num_states, infer_len  = xn[0][0][0].shape
        xn_array = np.zeros( (num_itr, num_states, infer_len, num_policies) )
        for policy_i in range(num_policies):
            for itr in range(num_itr):
                xn_array[itr,:,:,policy_i] = xn[policy_i][itr][0] 
    
    return xn_array

def plot_beliefs(belief_dist, title=""):
    """
    Utility function that plots a bar chart of a categorical probability distribution,
    with each bar height corresponding to the probability of one of the elements of the categorical
    probability vector.
    """

    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title)
    plt.show()
    
def plot_likelihood(A, title=""):
    """
    Utility function that shows a heatmap of a 2-D likelihood (hidden causes in the columns, observations in the rows),
    with hotter colors indicating higher probability.
    """

    ax = sns.heatmap(A, cmap="OrRd", linewidth=2.5)
    plt.xticks(range(A.shape[1]+1))
    plt.yticks(range(A.shape[0]+1))
    plt.title(title)
    plt.show()


    
   

