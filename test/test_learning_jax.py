#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import unittest

import numpy as np
from jax import numpy as jnp, random as jr
import jax.tree_util as jtu

from pymdp import utils

from pymdp.legacy.learning import update_obs_likelihood_dirichlet as update_pA_numpy
from pymdp.legacy.learning import update_obs_likelihood_dirichlet_factorized as update_pA_numpy_factorized
from pymdp.learning import update_obs_likelihood_dirichlet as update_pA_jax
from pymdp.legacy import utils as legacy_utils

from pymdp.legacy.learning import update_state_likelihood_dirichlet as update_pB_numpy
from pymdp.legacy.learning import update_state_likelihood_dirichlet_interactions as update_pB_interactions_numpy

from pymdp.learning import update_state_transition_dirichlet as update_pB_jax


def _to_numpy_list_of_arrs(jax_tree):
    """ Helper function to convert a JAX pytree (list of arrays) to a list of numpy arrays, before casting to object array in legacy_utils.obj_array_from_list"""
    return jtu.tree_map(lambda x: np.array(x), jax_tree)

class TestLearningJax(unittest.TestCase):

    def test_update_observation_likelihood_fullyconnected(self):
        """
        Testing JAX-ified version of updating Dirichlet posterior over observation likelihood parameters (qA is posterior, pA is prior, and A is expectation
        of likelihood wrt to current posterior over A, i.e. $A = E_{Q(A)}[P(o|s,A)]$.

        This is the so-called 'fully-connected' version where all hidden state factors drive each modality (i.e. A_dependencies is a list of lists of hidden state factors)
        """

        num_obs_list = [[5], [10, 3, 2], [2, 4, 4, 2], [10]]
        num_states_list = [[2, 3, 4], [2], [4, 5], [3]]

        A_dependencies_list = [[[0, 1, 2]], [[0], [0], [0]], [[0, 1], [0, 1], [0, 1], [0, 1]], [[0]]]

        keys = jr.split(jr.PRNGKey(42), len(num_obs_list)*3).reshape((len(num_obs_list), 3, 2))
        for (keys_per_model, num_obs, num_states, A_dependencies) in zip(keys, num_obs_list, num_states_list, A_dependencies_list):

            # create jax arrays
            A_jax = utils.random_A_array(keys_per_model[1], num_obs, num_states, A_dependencies=A_dependencies)
            pA_jax = utils.list_array_scaled([a.shape for a in A_jax], scale=3.0)

            # convert to numpy arrays for validation
            A_np = legacy_utils.obj_array_from_list(A_jax)
            pA_np = legacy_utils.obj_array_from_list(pA_jax)
          
            # create random observations
            obs_np = legacy_utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs_np[m] = legacy_utils.onehot(np.random.randint(obs_dim), obs_dim)

            # create random state posterior
            qs_jax = utils.random_factorized_categorical(keys_per_model[2], num_states)
            qs_np = legacy_utils.obj_array_from_list(qs_jax)

            l_rate = 1.0

            # run numpy version of learning
            qA_np_test = update_pA_numpy(pA_np, A_np, obs_np, qs_np, lr=l_rate)

            obs_jax = jtu.tree_map(lambda x: jnp.array(x)[None], list(obs_np))
            qs_jax = jtu.tree_map(lambda x: jnp.expand_dims(x,0), qs_jax)

            qA_jax_test, _ = update_pA_jax(
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

        num_obs_list = [[5], [10, 3, 2], [2, 4, 4, 2], [10]]
        num_states_list = [[2, 3, 4], [2, 5, 2], [4, 5], [3]]

        A_dependencies_list = [[[0, 1]], [[0, 1], [1], [1, 2]], [[0, 1], [0], [0, 1], [1]], [[0]]]

        keys = jr.split(jr.PRNGKey(43), len(num_obs_list)*3).reshape((len(num_obs_list), 3, 2))
        for (keys_per_model, num_obs, num_states, A_dependencies) in zip(keys, num_obs_list, num_states_list, A_dependencies_list):
            
            # create jax arrays
            A_jax = utils.random_A_array(keys_per_model[1], num_obs, num_states, A_dependencies=A_dependencies)
            pA_jax = utils.list_array_scaled([a.shape for a in A_jax], scale=3.0)

            # convert to numpy arrays for validation
            A_np = legacy_utils.obj_array_from_list(A_jax)
            pA_np = legacy_utils.obj_array_from_list(pA_jax)

            # create random observations
            obs_np = legacy_utils.obj_array(len(num_obs))
            for m, obs_dim in enumerate(num_obs):
                obs_np[m] = legacy_utils.onehot(np.random.randint(obs_dim), obs_dim)

            # create random state posterior
            qs_jax = utils.random_factorized_categorical(keys_per_model[2], num_states)
            qs_np = legacy_utils.obj_array_from_list(qs_jax)
            l_rate = 1.0

            # run numpy version of learning
            qA_np_test = update_pA_numpy_factorized(pA_np, A_np, obs_np, qs_np, A_dependencies, lr=l_rate)

            obs_jax = jtu.tree_map(lambda x: jnp.array(x)[None], list(obs_np))
            qs_jax = jtu.tree_map(lambda x: jnp.expand_dims(x, 0), qs_jax)

            qA_jax_test, _ = update_pA_jax(
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

    def test_update_state_likelihood_single_factor_no_actions(self):
        """
        Testing the JAXified version of updating Dirichlet posterior over transition likelihood parameters.
        qB is the posterior, pB is the prior and B is the expectation of the likelihood wrt the
        current posterior over B, i.e. $B = E_Q(B)[P(s_t | s_{t-1}, u_{t-1}, B)]
        """

        num_states = [3]
        num_controls = [1]

        l_rate = 1.0

        qs_prev_key, qs_key, b_key = jr.split(jr.PRNGKey(0), 3)
        # Create random variables to run the update on
        qs_prev_jax = utils.random_factorized_categorical(qs_prev_key, num_states)
        qs_jax = utils.random_factorized_categorical(qs_key, num_states)
        B_jax = utils.random_B_array(b_key, num_states, num_controls)
        pB_jax = utils.list_array_scaled([B_f.shape for B_f in B_jax], scale=1.0)

        qs_prev_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_prev_jax))
        qs_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_jax))
        B_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(B_jax))
        pB_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(pB_jax))

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated_numpy = update_pB_numpy(pB_np, B_np, action, qs_np, qs_prev_np, lr=l_rate, factors="all")

        action_jax = jnp.array([action])

        belief_jax = jtu.tree_map(
            lambda x, y: [x[None, ...], y[None, ...]], qs_jax, qs_prev_jax
        )

        pB_updated_jax, _ = update_pB_jax(pB_jax, B_jax, belief_jax, action_jax, num_controls=num_controls, lr=l_rate)

        for pB_np, pB_jax in zip(pB_updated_numpy, pB_updated_jax):
            self.assertTrue(pB_np.shape == pB_jax.shape)
            self.assertTrue(np.allclose(pB_np, pB_jax))

    def test_update_state_likelihood_single_factor_with_actions(self):
        """
        Testing the JAXified version of updating Dirichlet posterior over transition likelihood parameters.
        qB is the posterior, pB is the prior and B is the expectation of the likelihood wrt the
        current posterior over B, i.e. $B = E_Q(B)[P(s_t | s_{t-1}, u_{t-1}, B)]
        """

        num_states = [3]
        num_controls = [3]

        l_rate = 1.0

        qs_prev_key, qs_key, b_key = jr.split(jr.PRNGKey(1), 3)
        # Create random variables to run the update on
        qs_prev_jax = utils.random_factorized_categorical(qs_prev_key, num_states)
        qs_jax = utils.random_factorized_categorical(qs_key, num_states)
        B_jax = utils.random_B_array(b_key, num_states, num_controls)
        pB_jax = utils.list_array_scaled([B_f.shape for B_f in B_jax], scale=1.0)

        qs_prev_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_prev_jax))
        qs_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_jax))
        B_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(B_jax))
        pB_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(pB_jax))

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated_numpy = update_pB_numpy(pB_np, B_np, action, qs_np, qs_prev_np, lr=l_rate, factors="all")

        action_jax = jnp.array([action])

        belief_jax = jtu.tree_map(
            lambda x, y: [x[None, ...], y[None, ...]], qs_jax, qs_prev_jax
        )

        pB_updated_jax, _ = update_pB_jax(pB_jax, B_jax, belief_jax, action_jax, num_controls=num_controls, lr=l_rate)

        for pB_np, pB_jax in zip(pB_updated_numpy, pB_updated_jax):
            self.assertTrue(pB_np.shape == pB_jax.shape)
            self.assertTrue(np.allclose(pB_np, pB_jax))

    def test_update_state_likelihood_multi_factor_all_factors_no_actions(self):
        """
        Testing the JAXified version of updating Dirichlet posterior over transition likelihood parameters.
        qB is the posterior, pB is the prior and B is the expectation of the likelihood wrt the
        current posterior over B, i.e. $B = E_Q(B)[P(s_t | s_{t-1}, u_{t-1}, B)]$
        """

        num_states = [3, 4]
        num_controls = [1, 1]
        l_rate = 1.0

        qs_prev_key, qs_key, b_key = jr.split(jr.PRNGKey(2), 3)
        # Create random variables to run the update on
        qs_prev_jax = utils.random_factorized_categorical(qs_prev_key, num_states)
        qs_jax = utils.random_factorized_categorical(qs_key, num_states)
        B_jax = utils.random_B_array(b_key, num_states, num_controls)
        pB_jax = utils.list_array_scaled([B_f.shape for B_f in B_jax], scale=1.0)

        qs_prev_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_prev_jax))
        qs_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_jax))
        B_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(B_jax))
        pB_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(pB_jax))

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated_numpy = update_pB_numpy(pB_np, B_np, action, qs_np, qs_prev_np, lr=l_rate, factors="all")

        action_jax = jnp.array([action])

        belief_jax = jtu.tree_map(
            lambda x, y: [x[None, ...], y[None, ...]], qs_jax, qs_prev_jax
        )

        pB_updated_jax, _ = update_pB_jax(pB_jax, B_jax, belief_jax, action_jax, num_controls=num_controls, lr=l_rate)

        for pB_np, pB_jax in zip(pB_updated_numpy, pB_updated_jax):
            self.assertTrue(pB_np.shape == pB_jax.shape)
            self.assertTrue(np.allclose(pB_np, pB_jax))

    def test_update_state_likelihood_multi_factor_all_factors_with_actions(self):
        """
        Testing the JAXified version of updating Dirichlet posterior over transition likelihood parameters.
        qB is the posterior, pB is the prior and B is the expectation of the likelihood wrt the
        current posterior over B, i.e. $B = E_Q(B)[P(s_t | s_{t-1}, u_{t-1}, B)]$
        """
        num_states = [3, 4]
        num_controls = [3, 5]
        
        l_rate = 1.0
        qs_prev_key, qs_key, b_key = jr.split(jr.PRNGKey(3), 3)
        # Create random variables to run the update on
        qs_prev_jax = utils.random_factorized_categorical(qs_prev_key, num_states)
        qs_jax = utils.random_factorized_categorical(qs_key, num_states)
        B_jax = utils.random_B_array(b_key, num_states, num_controls)
        pB_jax = utils.list_array_scaled([B_f.shape for B_f in B_jax], scale=1.0)

        qs_prev_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_prev_jax))
        qs_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_jax))
        B_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(B_jax))
        pB_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(pB_jax))

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated_numpy = update_pB_numpy(pB_np, B_np, action, qs_np, qs_prev_np, lr=l_rate, factors="all")
        action_jax = jnp.array([action])

        belief_jax = jtu.tree_map(
            lambda x, y: [x[None, ...], y[None, ...]], qs_jax, qs_prev_jax
        )

        pB_updated_jax, _ = update_pB_jax(pB_jax, B_jax, belief_jax, action_jax, num_controls=num_controls, lr=l_rate)

        for pB_np, pB_jax in zip(pB_updated_numpy, pB_updated_jax):
            self.assertTrue(pB_np.shape == pB_jax.shape)
            self.assertTrue(np.allclose(pB_np, pB_jax))

    def test_update_state_likelihood_multi_factor_some_factors_no_action(self):
        """
        Testing the JAXified version of updating Dirichlet posterior over transition likelihood parameters.
        qB is the posterior, pB is the prior and B is the expectation of the likelihood wrt the
        current posterior over B, i.e. $B = E_Q(B)[P(s_t | s_{t-1}, u_{t-1}, B)]$
        """

        num_states = [3, 4, 2]
        num_controls = [3, 5, 5]
        
        l_rate = 1.0

        qs_prev_key, qs_key, b_key = jr.split(jr.PRNGKey(4), 3)
        # Create random variables to run the update on
        qs_prev_jax = utils.random_factorized_categorical(qs_prev_key, num_states)
        qs_jax = utils.random_factorized_categorical(qs_key, num_states)
        B_jax = utils.random_B_array(b_key, num_states, num_controls)
        pB_jax = utils.list_array_scaled([B_f.shape for B_f in B_jax], scale=1.0)

        qs_prev_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_prev_jax))
        qs_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_jax))
        B_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(B_jax))
        pB_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(pB_jax))

        action = list(np.array([np.random.randint(c_dim) for c_dim in num_controls]))

        factors_to_update = np.random.choice(list(range(len(B_jax))), replace=False, size=(2,)).tolist()

        pB_updated_numpy = update_pB_numpy(pB_np, B_np, action, qs_np, qs_prev_np, lr=l_rate, factors=factors_to_update)

        belief_jax = jtu.tree_map(
            lambda x, y: [x[None, ...], y[None, ...]], qs_jax, qs_prev_jax
        )

        action_jax = jnp.array([action])

        pB_updated_jax_factors, _ = update_pB_jax(
            pB_jax, B_jax, belief_jax, action_jax, num_controls=num_controls, lr=l_rate, factors_to_update=factors_to_update
        )

        for pB_np, pB_jax in zip(pB_updated_numpy, pB_updated_jax_factors):
            self.assertTrue(pB_np.shape == pB_jax.shape)
            self.assertTrue(np.allclose(pB_np, pB_jax))

    def test_update_state_likelihood_with_interactions(self):
        """
        Test for `learning.update_state_likelihood_dirichlet_factorized`, which is the learning function updating prior
        Dirichlet parameters over the transition likelihood (pB) in the case that there are allowable interactions
        between hidden state factors, i.e. the dynamics of factor `f` may depend on more than just its control factor
        and its own state.
        """

        """ Test version with interactions """
        num_states = [3, 4, 5]
        num_controls = [2, 1, 1]
        B_dependencies = [[0, 1], [0, 1, 2], [1, 2]]

        l_rate = jr.uniform(jr.PRNGKey(123), (), minval=0.01, maxval=1.0)  # sample some positive learning rate

        qs_prev_key, qs_key, b_key = jr.split(jr.PRNGKey(5), 3)
        # Create random variables to run the update on
        qs_prev_jax = utils.random_factorized_categorical(qs_prev_key, num_states)
        qs_jax = utils.random_factorized_categorical(qs_key, num_states)
        B_jax = utils.random_B_array(b_key, num_states, num_controls, B_dependencies=B_dependencies)
        pB_jax = utils.list_array_scaled([B_f.shape for B_f in B_jax], scale=1.0)

        qs_prev_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_prev_jax))
        qs_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(qs_jax))
        B_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(B_jax))
        pB_np = legacy_utils.obj_array_from_list(_to_numpy_list_of_arrs(pB_jax))
       
        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated_numpy = update_pB_interactions_numpy(
            pB_np, B_np, action, qs_np, qs_prev_np, B_factor_list=B_dependencies, lr=l_rate, factors="all"
        )

        action_jax = jnp.array([action])

        belief_jax = []
        for f, deps in enumerate(B_dependencies):
            q_f = qs_jax[f][None, ...]                                  # shape (1, n_f)
            q_prev_list = [qs_prev_jax[fi][None, ...] for fi in deps]   # one per dependency
            belief_jax.append([q_f, *q_prev_list])

        pB_updated_jax, _ = update_pB_jax(pB_jax, B_jax, belief_jax, action_jax, lr=l_rate, num_controls=num_controls)

        for pB_np, pB_jax in zip(pB_updated_numpy, pB_updated_jax):
            self.assertTrue(pB_np.shape == pB_jax.shape)
            self.assertTrue(np.allclose(pB_np, pB_jax))


if __name__ == "__main__":
    unittest.main()
