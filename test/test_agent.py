import os
import unittest

import numpy as np
from pymdp.agent import Agent
from pymdp.core.utils import random_A_matrix, random_B_matrix, obj_array_zeros, get_model_dimensions, convert_observation_array

DATA_PATH = "test/matlab_crossval/output/"

class TestAgent(unittest.TestCase):
   
    def test_fpi_inference(self):
        num_obs = [2, 4]
        num_states = [2, 2]
        num_control = [2, 2]
        A = random_A_matrix(num_obs, num_states)
        B = random_B_matrix(num_states, num_control)

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
        A = random_A_matrix(num_obs, num_states)
        B = random_B_matrix(num_states, num_control)

        C = utils.obj_array_zeros(num_obs)
        C[1][0] = 1.0  
        C[1][1] = -2.0  

        agent = Agent(A=A, B=B, C=C, control_fac_idx=[1], inference_algo="MMP", policy_len=5, inference_horizon=1)
        o = [0, 2]
        qx = agent.infer_states(o)

        print(qx[0].shape)
        print(qx[1].shape)

    
    def test_mmp_active_inference(self):
        """
        Tests to make sure whole active inference loop works (with various past and future
        inference/policy horizons).
        """

        num_obs = [3, 2]
        num_states = [4, 3]
        num_control = [1, 3]
        A = random_A_matrix(num_obs, num_states)
        B = random_B_matrix(num_states, num_control)

        C = obj_array_zeros(num_obs)
        C[1][0] = 1.0  
        C[1][1] = -2.0  

        agent = Agent(A=A, B=B, C=C, control_fac_idx=[1], inference_algo="MMP", policy_len=2, inference_horizon=3)

        T = 10

        for t in range(T):

            o = [np.random.randint(num_ob) for num_ob in num_obs] # just randomly generate observations at each timestep, no generative process
            qx = agent.infer_states(o)
            agent.infer_policies()
            action = agent.sample_action()
        
        print(agent.prev_actions)
        print(agent.prev_obs)
    
    def test_active_inference_SPM_a(self):
        """
        Test against output of SPM_MDP_VB_X Case A - one hidden state factor, one observation modality, policy_len = 1
        """

        

    def test_active_inference_SPM_b(self):
        """
        @ TODO
        """
        pass

if __name__ == "__main__":
    unittest.main()