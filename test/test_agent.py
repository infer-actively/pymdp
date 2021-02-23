import os
import unittest

import numpy as np
from pymdp.agent import Agent
from pymdp.core import utils


class TestAgent(unittest.TestCase):
   
    def test_fpi_inference(self):
        num_obs = [2, 4]
        num_states = [2, 2]
        num_control = [2, 2]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_control)

        C = utils.obj_array_zeros([num_ob for num_ob in num_obs])
        C[1][0] = 1.0  
        C[1][1] = -2.0  

        agent = Agent(A=A, B=B, C=C, control_fac_idx=[1])
        o, s = [0, 2], [0, 0]
        qx = agent.infer_states(o)
        agent.infer_policies()
        action = agent.sample_action()

        self.assertEqual(len(action), len(num_control))

    def test_mmp_inference(self):
        num_obs = [2, 4]
        num_states = [2, 2]
        num_control = [2, 2]
        A = utils.random_A_matrix(num_obs, num_states)
        B = utils.random_B_matrix(num_states, num_control)

        C = utils.obj_array_zeros([num_ob for num_ob in num_obs])
        C[1][0] = 1.0  
        C[1][1] = -2.0  

        agent = Agent(A=A, B=B, C=C, control_fac_idx=[1], inference_algo="MMP", policy_len=5, inference_horizon=1)
        o = [0, 2]
        qx = agent.infer_states(o)

        print(qx[0].shape)
        print(qx[1].shape)

        #agent.infer_policies()
        # action = agent.sample_action()


if __name__ == "__main__":
    unittest.main()