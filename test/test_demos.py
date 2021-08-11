import unittest
import numpy as np
import copy

from pymdp.agent import Agent
from pymdp.core import utils, test_models
from pymdp.core.maths import softmax

class TestDemos(unittest.TestCase):

    def test_agent_demo(self):
        """
        This unit test just runs all the code in the agent_demo Jupyter notebook to make sure the code works after changes
        """

        A, B, C, control_fac_idx = test_models.generate_epistemic_MAB_model()

        num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A = A, B = C)
       
        agent = Agent(A=A, B=B, C=C, control_fac_idx=control_fac_idx)

        # transition/observation matrices characterising the generative process
        A_gp = copy.deepcopy(A)
        B_gp = copy.deepcopy(B)

        # initial state
        T = 20 # number of timesteps in the simulation
        observation = [2, 2, 0] # initial observation -- no evidence for which arm is rewarding, neutral reward observation, and see themselves in the starting state
        state = [0, 0] # initial (true) state -- the reward condition is highly rewarding, and the true position in the 'start' position

        action_history = []

        for t in range(T):
    
            # update agent
            belief_state = agent.infer_states(observation)
            agent.infer_policies()
            action = agent.sample_action()

            action_history.append(action)
            
            # update environment
            for f, s in enumerate(state):
                state[f] = utils.sample(B_gp[f][:, s, int(action[f])])

            for g, _ in enumerate(observation):
                observation[g] = utils.sample(A_gp[g][:, state[0], state[1]])
        
        # actions_matrix = np.stack(action_history, axis =0)

        # print(f'Proportion of time spent playing the arm: {(actions_matrix[:,1] == 1).sum(axis=0) / T}' )
        # print(f'Proportion of time spent sampling the arm: {(actions_matrix[:,1] == 2).sum(axis=0) / T}' )

        # make sure the first action is sampling
        self.assertEqual(action_history[0][-1], 2)

        # make sure the last action is playing
        self.assertEqual(action_history[-1][-1], 1)

    
if __name__ == "__main__":
    unittest.main()