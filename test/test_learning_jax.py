#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import unittest

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu

from pymdp.learning import update_obs_likelihood_dirichlet as update_pA_numpy
from pymdp.learning import update_obs_likelihood_dirichlet_factorized as update_pA_numpy_factorized
from pymdp.jax.learning import update_obs_likelihood_dirichlet as update_pA_jax
from pymdp import utils

class TestLearningJax(unittest.TestCase):

    def test_update_observation_likelihood_fullyconnected(self):
        """
        Testing JAX-ified version of updating Dirichlet posterior over observation likelihood parameters (qA is posterior, pA is prior, and A is expectation
        of likelihood wrt to current posterior over A, i.e. $A = E_{Q(A)}[P(o|s,A)]$.

        This is the so-called 'fully-connected' version where all hidden state factors drive each modality (i.e. A_dependencies is a list of lists of hidden state factors)
        """

        num_obs_list = [ [5], 
                        [10, 3, 2], 
                        [2, 4, 4, 2],
                        [10]
                        ]
        num_states_list = [ [2,3,4], 
                        [2], 
                        [4,5],
                        [3] 
                        ]

        A_dependencies_list = [ [ [0,1,2] ],
                                [ [0], [0], [0] ],
                                [ [0,1], [0,1], [0,1], [0,1] ],
                                [ [0] ]
                                ]

        for (num_obs, num_states, A_dependencies) in zip(num_obs_list, num_states_list, A_dependencies_list):
            # create numpy arrays to test numpy version of learning

            # create A matrix initialization (expected initial value of P(o|s, A)) and prior over A (pA)
            A_np = utils.random_A_matrix(num_obs, num_states)
            pA_np = utils.dirichlet_like(A_np, scale = 3.0)

            # create random observations 
            obs_np = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs_np[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            # create random state posterior
            qs_np = utils.random_single_categorical(num_states)

            l_rate = 1.0

            # run numpy version of learning
            qA_np_test = update_pA_numpy(pA_np, A_np, obs_np, qs_np, lr=l_rate)

            pA_jax = jtu.tree_map(lambda x: jnp.array(x), list(pA_np))
            A_jax = jtu.tree_map(lambda x: jnp.array(x), list(A_np))
            obs_jax = jtu.tree_map(lambda x: jnp.array(x)[None], list(obs_np))
            qs_jax = jtu.tree_map(lambda x: jnp.array(x)[None], list(qs_np))

            qA_jax_test, E_qA_jax_test = update_pA_jax(
                pA_jax,
                A_jax,
                obs_jax,
                qs_jax,
                A_dependencies=A_dependencies,
                onehot_obs=True,
                num_obs=num_obs,
                lr=l_rate
            )

            for modality, obs_dim in enumerate(num_obs):
                self.assertTrue(np.allclose(qA_jax_test[modality], qA_np_test[modality]))

    def test_update_observation_likelihood_factorized(self):
        """
        Testing JAX-ified version of updating Dirichlet posterior over observation likelihood parameters (qA is posterior, pA is prior, and A is expectation
        of likelihood wrt to current posterior over A, i.e. $A = E_{Q(A)}[P(o|s,A)]$.

        This is the factorized version where only some hidden state factors drive each modality (i.e. A_dependencies is a list of lists of hidden state factors)
        """

        num_obs_list = [ [5], 
                        [10, 3, 2], 
                        [2, 4, 4, 2],
                        [10]
                        ]
        num_states_list = [ [2,3,4], 
                        [2, 5, 2], 
                        [4,5],
                        [3] 
                        ]

        A_dependencies_list = [ [ [0,1] ],
                                [ [0, 1], [1], [1, 2] ],
                                [ [0,1], [0], [0,1], [1] ],
                                [ [0] ]
                                ]

        for (num_obs, num_states, A_dependencies) in zip(num_obs_list, num_states_list, A_dependencies_list):
            # create numpy arrays to test numpy version of learning

            # create A matrix initialization (expected initial value of P(o|s, A)) and prior over A (pA)
            A_np = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_dependencies)
            pA_np = utils.dirichlet_like(A_np, scale = 3.0)

            # create random observations 
            obs_np = utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs_np[m] = utils.onehot(np.random.randint(obs_dim), obs_dim)

            # create random state posterior
            qs_np = utils.random_single_categorical(num_states)

            l_rate = 1.0

            # run numpy version of learning
            qA_np_test = update_pA_numpy_factorized(pA_np, A_np, obs_np, qs_np, A_dependencies, lr=l_rate)

            pA_jax = jtu.tree_map(lambda x: jnp.array(x), list(pA_np))
            A_jax = jtu.tree_map(lambda x: jnp.array(x), list(A_np))
            obs_jax = jtu.tree_map(lambda x: jnp.array(x)[None], list(obs_np))
            qs_jax = jtu.tree_map(lambda x: jnp.array(x)[None], list(qs_np))

            qA_jax_test, E_qA_jax_test = update_pA_jax(
                pA_jax,
                A_jax,
                obs_jax,
                qs_jax,
                A_dependencies=A_dependencies,
                onehot_obs=True,
                num_obs=num_obs,
                lr=l_rate
            )

            for modality, obs_dim in enumerate(num_obs):
                self.assertTrue(np.allclose(qA_jax_test[modality],qA_np_test[modality]))

if __name__ == "__main__":
    unittest.main()