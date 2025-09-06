#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import unittest

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu

from pymdp.legacy.learning import update_obs_likelihood_dirichlet as update_pA_numpy
from pymdp.legacy.learning import update_obs_likelihood_dirichlet_factorized as update_pA_numpy_factorized
from pymdp.learning import update_obs_likelihood_dirichlet as update_pA_jax
from pymdp.legacy import utils

from pymdp.legacy.learning import update_state_likelihood_dirichlet as update_pB_numpy
from pymdp.legacy.learning import update_state_likelihood_dirichlet_interactions as update_pB_interactions_numpy

from pymdp.learning import update_state_transition_dirichlet as update_pB_jax


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

        for num_obs, num_states, A_dependencies in zip(num_obs_list, num_states_list, A_dependencies_list):
            # create numpy arrays to test numpy version of learning

            # create A matrix initialization (expected initial value of P(o|s, A)) and prior over A (pA)
            A_np = utils.random_A_matrix(num_obs, num_states)
            pA_np = utils.dirichlet_like(A_np, scale=3.0)

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

        num_obs_list = [[5], [10, 3, 2], [2, 4, 4, 2], [10]]
        num_states_list = [[2, 3, 4], [2, 5, 2], [4, 5], [3]]

        A_dependencies_list = [[[0, 1]], [[0, 1], [1], [1, 2]], [[0, 1], [0], [0, 1], [1]], [[0]]]

        for num_obs, num_states, A_dependencies in zip(num_obs_list, num_states_list, A_dependencies_list):
            # create numpy arrays to test numpy version of learning

            # create A matrix initialization (expected initial value of P(o|s, A)) and prior over A (pA)
            A_np = utils.random_A_matrix(num_obs, num_states, A_factor_list=A_dependencies)
            pA_np = utils.dirichlet_like(A_np, scale=3.0)

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

        # Create random variables to run the update on
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])
        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated_numpy = update_pB_numpy(pB, B, action, qs, qs_prev, lr=l_rate, factors="all")

        pB_jax = [jnp.array(b) for b in pB]

        action_jax = jnp.array([action])

        belief_jax = []
        for f in range(len(num_states)):
            # Extract factor
            q_f = jnp.array([qs[..., f].tolist()])
            q_prev_f = jnp.array([qs_prev[..., f].tolist()])
            belief_jax.append([q_f, q_prev_f])

        pB_updated_jax, _ = update_pB_jax(pB_jax, B, belief_jax, action_jax, num_controls=num_controls, lr=l_rate)

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

        # Create random variables to run the update on
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])
        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated_numpy = update_pB_numpy(pB, B, action, qs, qs_prev, lr=l_rate, factors="all")

        action_jax = jnp.array([action])

        belief_jax = []
        for f in range(len(num_states)):
            # Extract factor
            q_f = jnp.array([qs[..., f].tolist()])
            q_prev_f = jnp.array([qs_prev[..., f].tolist()])
            belief_jax.append([q_f, q_prev_f])

        pB_jax = [jnp.array(b) for b in pB]

        pB_updated_jax, _ = update_pB_jax(pB_jax, B, belief_jax, action_jax, num_controls=num_controls, lr=l_rate)

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
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated_numpy = update_pB_numpy(pB, B, action, qs, qs_prev, lr=l_rate, factors="all")

        action_jax = jnp.array([action])

        belief_jax = []
        for f in range(len(num_states)):
            # Extract factor
            q_f = jnp.array([qs[..., f].tolist()])
            q_prev_f = jnp.array([qs_prev[..., f].tolist()])
            belief_jax.append([q_f, q_prev_f])

        pB_jax = [jnp.array(b) for b in pB]

        pB_updated_jax, _ = update_pB_jax(pB_jax, B, belief_jax, action_jax, num_controls=num_controls, lr=l_rate)

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
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated_numpy = update_pB_numpy(pB, B, action, qs, qs_prev, lr=l_rate, factors="all")

        action_jax = jnp.array([action])

        belief_jax = []
        for f in range(len(num_states)):
            # Extract factor
            q_f = jnp.array([qs[..., f].tolist()])
            q_prev_f = jnp.array([qs_prev[..., f].tolist()])
            belief_jax.append([q_f, q_prev_f])

        pB_jax = [jnp.array(b) for b in pB]

        pB_updated_jax, _ = update_pB_jax(pB_jax, B, belief_jax, action_jax, num_controls=num_controls, lr=l_rate)

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
        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)
        l_rate = 1.0

        B = utils.random_B_matrix(num_states, num_controls)
        pB = utils.obj_array_ones([B_f.shape for B_f in B])

        action = list(np.array([np.random.randint(c_dim) for c_dim in num_controls]))

        factors_to_update = np.random.choice(list(range(len(B))), replace=False, size=(2,)).tolist()

        pB_updated_numpy = update_pB_numpy(pB, B, action, qs, qs_prev, lr=l_rate, factors=factors_to_update)

        belief_jax = []
        for f in range(len(num_states)):
            # Extract factor
            q_f = jnp.array([qs[..., f].tolist()])
            q_prev_f = jnp.array([qs_prev[..., f].tolist()])
            belief_jax.append([q_f, q_prev_f])

        pB_jax = [jnp.array(b) for b in pB]

        action_jax = jnp.array([action])

        pB_updated_jax_factors, _ = update_pB_jax(
            pB_jax, B, belief_jax, action_jax, num_controls=num_controls, lr=l_rate, factors_to_update=factors_to_update
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
        B_factor_list = [[0, 1], [0, 1, 2], [1, 2]]

        qs_prev = utils.random_single_categorical(num_states)
        qs = utils.random_single_categorical(num_states)

        B = utils.random_B_matrix(num_states, num_controls, B_factor_list=B_factor_list)
        pB = utils.dirichlet_like(B, scale=1.0)
        l_rate = np.random.rand()  # sample some positive learning rate

        action = np.array([np.random.randint(c_dim) for c_dim in num_controls])

        pB_updated_numpy = update_pB_interactions_numpy(
            pB, B, action, qs, qs_prev, B_factor_list, lr=l_rate, factors="all"
        )

        action_jax = jnp.array([action])

        belief_jax = []
        for f in range(len(num_states)):
            # Extract factor
            q_f = jnp.array([qs[..., f].tolist()])
            q_prev_f = [jnp.array([qs_prev[..., fi].tolist()]) for fi in B_factor_list[f]]
            belief_jax.append([q_f, *q_prev_f])

        pB_jax = [jnp.array(b) for b in pB]

        pB_updated_jax, _ = update_pB_jax(pB_jax, B, belief_jax, action_jax, lr=l_rate, num_controls=num_controls)

        for pB_np, pB_jax in zip(pB_updated_numpy, pB_updated_jax):
            self.assertTrue(pB_np.shape == pB_jax.shape)
            self.assertTrue(np.allclose(pB_np, pB_jax))


if __name__ == "__main__":
    unittest.main()
