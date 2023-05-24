#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import os
import unittest

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu

from pymdp.jax.algos import run_vanilla_fpi as fpi_jax
from pymdp.jax.algos import run_factorized_fpi as fpi_jax_factorized
from pymdp.algos import run_vanilla_fpi as fpi_numpy
from pymdp.algos import run_mmp as mmp_numpy
from pymdp.jax.algos import run_mmp as mmp_jax
from pymdp.jax.algos import run_vmp as vmp_jax
from pymdp import utils, maths

from typing import Any, List


blanket_dict = {} # @TODO: implement factorized likelihoods for marginal message passing

num_states = [3]
num_obs = [3]

A = [ jnp.broadcast_to(jnp.array([[0.5, 0.5, 0.], 
                                  [0.0,  0.0,  1.], 
                                  [0.5, 0.5, 0.]]
                            ), (2, 3, 3) )]

B = [ jnp.broadcast_to(jnp.array([[0.0, 0.75, 0.0],
                                  [0.0, 0.25, 1.0],
                                  [1.0, 0.0, 0.0]]
            ), (2, 3, 3))]

# for the single modality, a sequence over time of observations (one hot vectors)
obs = [
        jnp.broadcast_to(jnp.array([[1., 0., 0.], 
                                    [0., 1., 0.], 
                                    [0., 0., 1.],
                                    [1., 0., 0.]])[:, None], (4, 2, 3) )
                        ]

prior = [jnp.ones((2, 3)) / 3.]

for t in range(4):
    loc_obs = jtu.tree_map( lambda o: o[:t+1], obs)
    qs_out = vmp_jax(A, B, loc_obs, prior, blanket_dict, num_iter=16, tau=1.)
    print(qs_out[0][:,0,:].round(3))

for t in range(4):
    loc_obs = jtu.tree_map( lambda o: o[:t+1], obs)
    qs_out = mmp_jax(A, B, loc_obs, prior, blanket_dict, num_iter=16, tau=1.)
    print(qs_out[0][:,0,:].round(3))

# class TestMessagePassing(unittest.TestCase):

#     # def test_fixed_point_iteration(self):
#     #     num_states_list = [ 
#     #                      [2, 2, 5],
#     #                      [2, 2, 2],
#     #                      [4, 4]
#     #     ]

#     #     num_obs_list = [
#     #                     [5, 10],
#     #                     [4, 3, 2],
#     #                     [5, 10, 6]
#     #     ]

#     #     for (num_states, num_obs) in zip(num_states_list, num_obs_list):

#     #         # numpy version
#     #         prior = utils.random_single_categorical(num_states)
#     #         A = utils.random_A_matrix(num_obs, num_states)

#     #         obs = utils.obj_array(len(num_obs))
#     #         for m, obs_dim in enumerate(num_obs):
#     #             obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

#     #         qs_numpy = fpi_numpy(A, obs, num_obs, num_states, prior=prior, num_iter=16, dF=1.0, dF_tol=-1.0) # set dF_tol to negative number so numpy version of FPI never stops early due to convergence

#     #         # jax version
#     #         prior = [jnp.array(prior_f) for prior_f in prior]
#     #         A = [jnp.array(a_m) for a_m in A]
#     #         obs = [jnp.array(o_m) for o_m in obs]

#     #         qs_jax = fpi_jax(A, obs, prior, num_iter=16)

#     #         for f, _ in enumerate(qs_jax):
#     #             self.assertTrue(np.allclose(qs_numpy[f], qs_jax[f]))


#     # def test_fixed_point_iteration_factorized_fullyconnected(self):
#     #     """
#     #     Test the factorized version of `run_vanilla_fpi`, named `run_factorized_fpi`
#     #     with multiple hidden state factors and multiple observation modalities.
#     #     """

#     #     num_states_list = [ 
#     #                      [2, 2, 5],
#     #                      [2, 2, 2],
#     #                      [4, 4]
#     #     ]

#     #     num_obs_list = [
#     #                     [5, 10],
#     #                     [4, 3, 2],
#     #                     [5, 10, 6]
#     #     ]

#     #     for (num_states, num_obs) in zip(num_states_list, num_obs_list):

#     #         # initialize arrays in numpy version
#     #         prior = utils.random_single_categorical(num_states)
#     #         A = utils.random_A_matrix(num_obs, num_states)

#     #         obs = utils.obj_array(len(num_obs))
#     #         for m, obs_dim in enumerate(num_obs):
#     #             obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

#     #         # jax version
#     #         prior = [jnp.array(prior_f) for prior_f in prior]
#     #         A = [jnp.array(a_m) for a_m in A]
#     #         obs = [jnp.array(o_m) for o_m in obs]

#     #         factor_lists = len(num_obs) * [list(range(len(num_states)))]

#     #         qs_jax = fpi_jax(A, obs, prior, num_iter=16)
#     #         qs_jax_factorized = fpi_jax_factorized(A, obs, prior, factor_lists, num_iter=16)

#     #         for f, _ in enumerate(qs_jax):
#     #             self.assertTrue(np.allclose(qs_jax[f], qs_jax_factorized[f]))

#     # def test_fixed_point_iteration_factorized_sparsegraph(self):
#     #     """
#     #     Test the factorized version of `run_vanilla_fpi`, named `run_factorized_fpi`
#     #     with multiple hidden state factors and multiple observation modalities, and with sparse conditional dependence relationships between hidden states
#     #     and observation modalities
#     #     """
        
#     #     num_states = [3, 4]
#     #     num_obs = [3, 3, 5]

#     #     prior = utils.random_single_categorical(num_states)

#     #     obs = utils.obj_array(len(num_obs))
#     #     for m, obs_dim in enumerate(num_obs):
#     #         obs[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

#     #     A_factor_list = [[0], [1], [0, 1]] # modalities 0 and 1 only depend on factors 0 and 1, respectively
#     #     A_reduced = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_factor_list)

#     #     # jax version
#     #     prior_jax = [jnp.array(prior_f) for prior_f in prior]
#     #     A_reduced_jax = [jnp.array(a_m) for a_m in A_reduced]
#     #     obs_jax = [jnp.array(o_m) for o_m in obs]

#     #     qs_out = fpi_jax_factorized(A_reduced_jax, obs_jax, prior_jax, A_factor_list, num_iter=16)

#     #     A_full = utils.initialize_empty_A(num_obs, num_states)
#     #     for m, A_m in enumerate(A_full):
#     #         other_factors = list(set(range(len(num_states))) - set(A_factor_list[m])) # list of the factors that modality `m` does not depend on

#     #         # broadcast or tile the reduced A matrix (`A_reduced`) along the dimensions of corresponding to `other_factors`
#     #         expanded_dims = [num_obs[m]] + [1 if f in other_factors else ns for (f, ns) in enumerate(num_states)]
#     #         tile_dims = [1] + [ns if f in other_factors else 1 for (f, ns) in enumerate(num_states)]
#     #         A_full[m] = np.tile(A_reduced[m].reshape(expanded_dims), tile_dims)

#     #     # jax version
#     #     A_full_jax = [jnp.array(a_m) for a_m in A_full]

#     #     qs_validation = fpi_jax(A_full_jax, obs_jax, prior_jax, num_iter=16)

#     #     for qs_f_val, qs_f_out in zip(qs_validation, qs_out):
#     #         self.assertTrue(np.allclose(qs_f_val, qs_f_out))

#     def test_marginal_message_passing(self):

#         blanket_dict = {} # @TODO: implement factorized likelihoods for marginal message passing

#         num_states = [3]
#         num_obs = [3]

#         A = [ jnp.broadcast_to(jnp.array([[0.5, 0.5, 0.0], 
#                         [0.0, 0.0, 1.0], 
#                         [0.5, 0.5, 0.0]]
#                     ), (2, 3, 3) )]

#         B = [ jnp.broadcast_to(jnp.array([[0.0, 0.5, 0.0],
#                         [0.0, 0.5, 1.0],
#                         [1.0, 0.0, 0.0]]
#                     ), (2, 3, 3))]
        
#         # for the single modality, a sequence over time of observations (one hot vectors)
#         obs = [
#              jnp.broadcast_to(jnp.array([[1, 0, 0], 
#                         [0, 1, 0], 
#                         [0, 0, 1],
#                         [1, 0, 0]])[:, None], (4, 2, 3) )
#             ]
        
#         prior = [jnp.ones((2, 3)) / 3.]

#         qs_out = mmp_jax(A, B, obs, prior, blanket_dict, num_iter=1, tau=1.)

#         print(qs_out[0])
        
#         # for qs_f_val, qs_f_out in zip(qs_validation, qs_out):
#         #     self.assertTrue(np.allclose(qs_f_val, qs_f_out).all())

#     def test_vmp(self):
#         pass
        

# if __name__ == "__main__":
#     unittest.main()       








    
