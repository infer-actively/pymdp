#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Tests
__author__: Conor Heins
"""

import os
import unittest

import numpy as np
import jax.numpy as jnp
from jax import vmap, nn, random
from jax.tree_util import register_pytree_node_class

from pymdp.jax.maths import compute_log_likelihood_single_modality
from pymdp.jax.utils import norm_dist

class TestAgentJax(unittest.TestCase):

    def test_vmappable_agent_methods(self):

        dim, N = 5, 10
        sampling_key = random.PRNGKey(1)

        @register_pytree_node_class
        class BasicAgent(object):
            def __init__(self, A, B):
                self.A = A
                self.B = B
                self.qs = norm_dist(jnp.ones(dim))
            
            def tree_flatten(self):
                children = (self.A, self.B)
                aux_data = None
                return (children, aux_data)

            @vmap
            def infer_states(self, obs):
                qs = nn.softmax(compute_log_likelihood_single_modality(obs, self.A))
                self.qs = qs # @NOTE: weirdly, adding this line doesn't actually change self.qs. When you query self.qs afterwards it's just the same as it was initialized in `self.__init__()`
                return qs

            @classmethod
            def tree_unflatten(cls, aux_data, children):
                return cls(*children) 

        A_key, B_key, obs_key, test_key = random.split(sampling_key, 4)

        all_A = vmap(norm_dist)(random.uniform(A_key, shape = (N, dim, dim)))
        all_B = vmap(norm_dist)(random.uniform(B_key, shape = (N, dim, dim)))
        all_obs = vmap(nn.one_hot, (0, None))(random.choice(obs_key, dim, shape = (N,)), dim)

        my_agent = BasicAgent(all_A, all_B)

        all_qs = my_agent.infer_states(all_obs) 

        # validate that the method broadcasted properly
        for id_to_check in range(N):
            validation_qs = nn.softmax(compute_log_likelihood_single_modality(all_obs[id_to_check], all_A[id_to_check]))
            self.assertTrue(jnp.allclose(validation_qs, all_qs[id_to_check]))

if __name__ == "__main__":
    unittest.main()       








    
