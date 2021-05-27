import os
import unittest

import numpy as np
from scipy.io import loadmat

from pymdp.agent import Agent, build_belief_array, build_xn_vn_array
from pymdp.core.utils import random_A_matrix, random_B_matrix, obj_array_zeros, get_model_dimensions, convert_observation_array
from pymdp.core.utils import to_arr_of_arr, to_numpy
from pymdp.core import control

DATA_PATH = "test/matlab_crossval/output/"

class TestAgent(unittest.TestCase):
   
    def test_fpi_inference(self):
        num_obs = [2, 4]
        num_states = [2, 2]
        num_control = [2, 2]
        A = random_A_matrix(num_obs, num_states)
        B = random_B_matrix(num_states, num_control)

        C = obj_array_zeros([num_ob for num_ob in num_obs])
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

        C = obj_array_zeros(num_obs)
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
    
    def test_active_inference_SPM_1a(self):
        """
        Test against output of SPM_MDP_VB_X.m
        1A - one hidden state factor, one observation modality, backwards horizon = 3, policy_len = 1, policy-conditional prior
        """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "vbx_test_1a.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"][0]
        B = mat_contents["B"][0]
        C = to_arr_of_arr(mat_contents["C"][0][0][:,0])
        obs_matlab = mat_contents["obs"].astype("int64")
        policy = mat_contents["policies"].astype("int64") - 1
        t_horizon = mat_contents["t_horizon"][0, 0].astype("int64")
        actions_matlab = mat_contents["actions"].astype("int64") - 1
        qs_matlab = mat_contents["qs"][0]
        xn_matlab = mat_contents["xn"][0]
        vn_matlab = mat_contents["vn"][0]

        likelihoods_matlab = mat_contents["likelihoods"][0]

        num_obs, num_states, _, num_factors = get_model_dimensions(A, B)
        obs = convert_observation_array(obs_matlab, num_obs)
        T = len(obs)

        agent = Agent(A=A, B=B, C=C, inference_algo="MMP", policy_len=1, 
                        inference_horizon=t_horizon, use_BMA = False, 
                        policy_sep_prior = True)
        
        actions_python = np.zeros(T)

        for t in range(T):
            o_t = (np.where(obs[t])[0][0],)
            qx, xn_t, vn_t = agent.infer_states_test(o_t)
            q_pi, efe= agent.infer_policies()
            action = agent.sample_action()

            actions_python[t] = action

            xn_python = build_xn_vn_array(xn_t)
            vn_python = build_xn_vn_array(vn_t)

            if t == T-1:
                xn_python = xn_python[:,:,:-1,:]
                vn_python = vn_python[:,:,:-1,:]

            start_tstep = max(0, agent.curr_timestep - agent.inference_horizon)
            end_tstep = min(agent.curr_timestep + agent.policy_len, T)

            xn_validation = xn_matlab[0][:,:,start_tstep:end_tstep,t,:]
            vn_validation = vn_matlab[0][:,:,start_tstep:end_tstep,t,:]

            self.assertTrue(np.isclose(xn_python, xn_validation).all())
            self.assertTrue(np.isclose(vn_python, vn_validation).all())
        
        self.assertTrue(np.isclose(actions_matlab[0,:],actions_python[:-1]).all())

    def test_active_inference_SPM_1b(self):
       """
        Test against output of SPM_MDP_VB_X.m
        1B - one hidden state factor, one observation modality, backwards horizon = 3, policy_len = 1, policy-conditional prior
        """
        

if __name__ == "__main__":
    unittest.main()