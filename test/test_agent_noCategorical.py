import os
import unittest

import numpy as np
from scipy.io import loadmat

from pymdp.agent import Agent, build_belief_array, build_xn_vn_array
from pymdp.core.utils import random_A_matrix, random_B_matrix, obj_array_zeros, get_model_dimensions, convert_observation_array
from pymdp.core.utils import to_arr_of_arr, to_numpy
from pymdp.core import control

class TestAgent_noCat(unittest.TestCase):
   
    def test_agent_init(self):
        num_obs = [2, 4]
        num_states = [2, 2]
        num_control = [2, 2]
        A = random_A_matrix(num_obs, num_states)
        B = random_B_matrix(num_states, num_control)

        C = obj_array_zeros([num_ob for num_ob in num_obs])
        C[1][0] = 1.0  
        C[1][1] = -2.0  

        agent = Agent(A=A, B=B, C=C, control_fac_idx=[1])

        print(agent.num_controls)

        print(agent.policies)

if __name__ == "__main__":
    unittest.main()

    