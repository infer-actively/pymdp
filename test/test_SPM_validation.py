import os
import unittest

import numpy as np
from scipy.io import loadmat

from pymdp.agent import Agent
from pymdp.utils import to_obj_array, build_xn_vn_array, get_model_dimensions, convert_observation_array
from pymdp.maths import dirichlet_log_evidence

DATA_PATH = "test/matlab_crossval/output/"

class TestSPM(unittest.TestCase):
    
    def test_active_inference_SPM_1a(self):
        """
        Test against output of SPM_MDP_VB_X.m
        1A - one hidden state factor, one observation modality, backwards horizon = 3, policy_len = 1, policy-conditional prior
        """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "vbx_test_1a.mat")
        mat_contents = loadmat(file_name=array_path)

        A = mat_contents["A"][0]
        B = mat_contents["B"][0]
        C = to_obj_array(mat_contents["C"][0][0][:,0])
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
            qx, xn_t, vn_t = agent._infer_states_test(o_t)
            q_pi, G= agent.infer_policies()
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

    def test_BMR_SPM_a(self):
        """
        Validate output of pymdp's `dirichlet_log_evidence` function 
        against output of `spm_MDP_log_evidence` from DEM in SPM (MATLAB)
        Test `a` tests the log evidence calculations across for a single
        reduced model, stored in a vector `r_dir`
        """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "bmr_test_a.mat")
        mat_contents = loadmat(file_name=array_path)
        F_valid = mat_contents["F"]

        # create BMR example from MATLAB
        x = np.linspace(1, 32, 128)

        p_dir    = np.ones(2)
        r_dir    = p_dir.copy()
        r_dir[1] = 8.

        F_out = np.zeros( (len(x), len(x)) )
        for i in range(len(x)):
            for j in range(len(x)):
                q_dir = np.array([x[i], x[j]])
                F_out[i,j] = dirichlet_log_evidence(q_dir, p_dir, r_dir)[0]

        self.assertTrue(np.allclose(F_valid, F_out))

    def test_BMR_SPM_b(self):
        """
        Validate output of pymdp's `dirichlet_log_evidence` function 
        against output of `spm_MDP_log_evidence` from DEM in SPM (MATLAB). 
        Test `b` vectorizes the log evidence calculations across a _matrix_ of 
        reduced models, with one reduced model prior per column of the argument `r_dir`
        """
        array_path = os.path.join(os.getcwd(), DATA_PATH + "bmr_test_b.mat")
        mat_contents = loadmat(file_name=array_path)
        F_valid = mat_contents["F"]
        s_dir_valid = mat_contents['s_dir']
        q_dir = mat_contents["q_dir"]
        p_dir = mat_contents["p_dir"]
        r_dir = mat_contents["r_dir"]
        
        F_out, s_dir_out = dirichlet_log_evidence(q_dir, p_dir, r_dir)

        self.assertTrue(np.allclose(F_valid, F_out))

        self.assertTrue(np.allclose(s_dir_valid, s_dir_out))



if __name__ == "__main__":
    unittest.main()