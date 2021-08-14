import unittest
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt

from pymdp.agent import Agent
from pymdp.core import utils, maths, default_models
from pymdp.envs import TMazeEnv, TMazeEnvNullOutcome

class TestDemos(unittest.TestCase):

    def test_agent_demo(self):
        """
        This unit test runs a more concise version of the
        code in the `agent_demo.ipynb` tutorial Jupyter notebook and the `agent_demo` Python script 
        to make sure the code works whenever we change something.
        """

        A, B, C, control_fac_idx = default_models.generate_epistemic_MAB_model()

        num_obs, num_states, num_modalities, num_factors = utils.get_model_dimensions(A = A, B = C)
       
        agent = Agent(A=A, B=B, C=C, control_fac_idx=control_fac_idx)

        # transition/observation matrices characterising the generative process
        A_gp = copy.deepcopy(A)
        B_gp = copy.deepcopy(B)

        # initial state
        T = 5 # number of timesteps in the simulation
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
        
        # make sure the first action is sampling
        self.assertEqual(action_history[0][-1], 2)

        # make sure the last action is playing
        self.assertEqual(action_history[-1][-1], 1)
    
    def test_tmaze_demo(self):
        """
        This unit test runs the a concise version of the code in the `tmaze_demo.ipynb` tutorial notebook to make sure it works if things are changed
        """

        def plot_beliefs(belief_dist, title=""):
            plt.grid(zorder=0)
            plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
            plt.xticks(range(belief_dist.shape[0]))
            plt.title(title)
            
        def plot_likelihood(A, title=""):
            ax = sns.heatmap(A, cmap="OrRd", linewidth=2.5)
            plt.xticks(range(A.shape[1]))
            plt.yticks(range(A.shape[0]))
            plt.title(title)

        reward_probabilities = [0.98, 0.02] # probabilities used in the original SPM T-maze demo
        env = TMazeEnv(reward_probs = reward_probabilities)

        '''test plotting of the observation likelihood (just plot one slice)'''
        A_gp = env.get_likelihood_dist()
        plot_likelihood(A_gp[1][:,:,0],'Reward Right')

        '''test plotting of the transition likelihood (just plot one slice)'''
        B_gp = env.get_transition_dist()
        plot_likelihood(B_gp[1][:,:,0],'Reward Condition Transitions')

        A_gm = copy.deepcopy(A_gp) # make a copy of the true observation likelihood to initialize the observation model
        B_gm = copy.deepcopy(B_gp)# make a copy of the true transition likelihood to initialize the transition model
        
        control_fac_idx = [0]
        agent = Agent(A=A_gm, B=B_gm, control_fac_idx=control_fac_idx)
        plot_beliefs(agent.D[0],"Beliefs about initial location")

        agent.C[1][1] = 3.0 # they like reward
        agent.C[1][2] = -3.0 # they don't like loss

        T = 5 # number of timesteps

        obs = env.reset() # reset the environment and get an initial observation

        # these are useful for displaying read-outs during the loop over time
        reward_conditions = ["Right", "Left"]
        location_observations = ['CENTER','RIGHT ARM','LEFT ARM','CUE LOCATION']
        reward_observations = ['No reward','Reward!','Loss!']
        cue_observations = ['Cue Right','Cue Left']
      
        for t in range(T):
            qx = agent.infer_states(obs)

            q_pi, efe = agent.infer_policies()

            action = agent.sample_action()

            obs = env.step(action)

            if int(action[0]) == 3:
                
                # if the reward condition is Reward on RIGHT
                if env.reward_condition == 0:
                    self.assertEqual(obs[2], 0) # this tests that the cue observation is 'Cue Right' in case of 'Reward on Right' condition

                # if the reward condition is Reward on RIGHT
                if env.reward_condition == 1:
                    self.assertEqual(obs[2], 1) # this tests that the cue observation is 'Cue Left' in case of 'Reward on Left' condition

            
        plot_beliefs(qx[1],"Final posterior beliefs about reward condition")
    
    def test_tmaze_learning_demo(self):
        """
        This unit test runs the a concise version of the code in the `tmaze_demo_learning.ipynb` tutorial notebook to make sure it works if things are changed
        """

        def plot_beliefs(belief_dist, title=""):
            plt.grid(zorder=0)
            plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
            plt.xticks(range(belief_dist.shape[0]))
            plt.title(title)
            
        def plot_likelihood(A, title=""):
            ax = sns.heatmap(A, cmap="OrRd", linewidth=2.5)
            plt.xticks(range(A.shape[1]))
            plt.yticks(range(A.shape[0]))
            plt.title(title)
        
        reward_probabilities = [0.85, 0.15] # the 'true' reward probabilities 
        env = TMazeEnvNullOutcome(reward_probs = reward_probabilities)
        A_gp = env.get_likelihood_dist()
        B_gp = env.get_transition_dist()

        pA = utils.dirichlet_like(A_gp, scale = 1e16)

        pA[1][1:,1:3,:] = 1.0

        A_gm = utils.norm_dist_obj_arr(pA) 

        B_gm = copy.deepcopy(B_gp)

        controllable_indices = [0] # this is a list of the indices of the hidden state factors that are controllable
        learnable_modalities = [1] # this is a list of the modalities that you want to be learn-able 

        agent = Agent(A=A_gm,pA=pA,B=B_gm,
              control_fac_idx=controllable_indices,
              modalities_to_learn=learnable_modalities,
              lr_pA = 0.25,
              use_param_info_gain=True)

        agent.D[0] = utils.onehot(0, agent.num_states[0])
        agent.C[1][1] = 2.0
        agent.C[1][2] = -2.0

        T = 1000 # number of timesteps

        obs = env.reset() # reset the environment and get an initial observation

        for t in range(T):
            agent.infer_states(obs)
            agent.infer_policies()
            action = agent.sample_action()
            agent.update_A(obs)     
            obs = env.step(action)
        
        # make sure they are learning the reward contingencies in the right general direction

        REWARD_ON_RIGHT = 0
        REWARD_ON_LEFT = 1

        REWARD = 1
        PUNISHMENT = 2

        RIGHT_ARM = 1
        LEFT_ARM = 2

        # in case the reward condition is 'Reward on RIGHT' 

        if env.reward_condition == REWARD_ON_RIGHT:

            prob_reward_right = agent.A[1][REWARD,RIGHT_ARM,REWARD_ON_RIGHT]
            prob_punishment_right = agent.A[1][PUNISHMENT,RIGHT_ARM,REWARD_ON_RIGHT]

            self.assertGreater(prob_reward_right, prob_punishment_right)

        # in case the reward condition is 'Reward on LEFT' 

        elif env.reward_condition == REWARD_ON_LEFT:

            prob_reward_left = agent.A[1][REWARD,LEFT_ARM,REWARD_ON_LEFT]
            prob_punishment_left = agent.A[1][PUNISHMENT,LEFT_ARM,REWARD_ON_LEFT]
            self.assertGreater(prob_reward_left, prob_punishment_left)

    def test_gridworld_genmodel_construction(self):
        """
        This unit test runs the a concise version of the code in the `gridworld_tutorial_1.ipynb` tutorial notebook to make sure it works if things are changed
        """
        


           
if __name__ == "__main__":
    unittest.main()