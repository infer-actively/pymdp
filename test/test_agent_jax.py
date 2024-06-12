#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Dimitrije Markovic, Conor Heins
"""

import os
import sys
import unittest

import jax.numpy as jnp
from jax import vmap, nn, random

# import the library directly from local source (rather than relying on the library being installed)
# insert the dependency so it's prioritized over an installed variant
sys.path.insert(0, os.path.abspath('../pymdp'))

from pymdp.jax.maths import compute_log_likelihood_single_modality
from pymdp.jax.utils import norm_dist
from equinox import Module

class TestAgentJax(unittest.TestCase):

    def test_vmappable_agent_methods(self):

        dim, N = 5, 10
        sampling_key = random.PRNGKey(1)

        class BasicAgent(Module):
            A: jnp.ndarray
            B: jnp.ndarray 
            qs: jnp.ndarray

            def __init__(self, A, B, qs=None):
                self.A = A
                self.B = B
                self.qs = jnp.ones((N, dim))/dim if qs is None else qs
            
            @vmap
            def infer_states(self, obs):
                qs = nn.softmax(compute_log_likelihood_single_modality(obs, self.A))
                return qs, BasicAgent(self.A, self.B, qs=qs)

        A_key, B_key, obs_key, test_key = random.split(sampling_key, 4)

        all_A = vmap(norm_dist)(random.uniform(A_key, shape = (N, dim, dim)))
        all_B = vmap(norm_dist)(random.uniform(B_key, shape = (N, dim, dim)))
        all_obs = vmap(nn.one_hot, (0, None))(random.choice(obs_key, dim, shape = (N,)), dim)

        my_agent = BasicAgent(all_A, all_B)

        all_qs, my_agent = my_agent.infer_states(all_obs)

        assert all_qs.shape == my_agent.qs.shape
        self.assertTrue(jnp.allclose(all_qs, my_agent.qs))

        # validate that the method broadcasted properly
        for id_to_check in range(N):
            validation_qs = nn.softmax(compute_log_likelihood_single_modality(all_obs[id_to_check], all_A[id_to_check]))
            self.assertTrue(jnp.allclose(validation_qs, all_qs[id_to_check]))

if __name__ == "__main__":
    unittest.main()       








    
