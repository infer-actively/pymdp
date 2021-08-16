import unittest
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt

from pymdp.agent import Agent
from pymdp import utils, maths, default_models
from pymdp import control
from pymdp.envs import TMazeEnv, TMazeEnvNullOutcome
from copy import deepcopy

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
        # self.assertEqual(action_history[0][-1], 2) #  @NOTE: Stochastic sampling means this is not always true!!!

        # make sure the last action is playing
        # self.assertEqual(action_history[-1][-1], 1) # @NOTE: Stochastic sampling means this is not always true!!!
    
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

        state_mapping = {0: (0,0), 1: (1,0), 2: (2,0), 3: (0,1), 4: (1,1), 5:(2,1), 6: (0,2), 7:(1,2), 8:(2,2)}

        grid = np.zeros((3,3))
        for linear_index, xy_coordinates in state_mapping.items():
            x, y = xy_coordinates
            grid[y,x] = linear_index # rows are the y-coordinate, columns are the x-coordinate -- so we index into the grid we'll be visualizing using '[y, x]'
        fig = plt.figure(figsize = (3,3))
        sns.set(font_scale=1.5)
        sns.heatmap(grid, annot=True,  cbar = False, fmt='.0f', cmap='crest')

        A = np.eye(9)

        labels = [state_mapping[i] for i in range(A.shape[1])]
        def plot_likelihood(A):
            fig = plt.figure(figsize = (6,6))
            ax = sns.heatmap(A, xticklabels = labels, yticklabels = labels, cbar = False)
            plt.title("Likelihood distribution (A)")

        plot_likelihood(A)

        P = {}
        dim = 3
        actions = {'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'STAY':4}

        for state_index, xy_coordinates in state_mapping.items():
            P[state_index] = {a : [] for a in range(len(actions))}
            x, y = xy_coordinates

            '''if your y-coordinate is all the way at the top (i.e. y == 0), you stay in the same place -- otherwise you move one upwards (achieved by subtracting 3 from your linear state index'''
            P[state_index][actions['UP']] = state_index if y == 0 else state_index - dim 

            '''f your x-coordinate is all the way to the right (i.e. x == 2), you stay in the same place -- otherwise you move one to the right (achieved by adding 1 to your linear state index)'''
            P[state_index][actions["RIGHT"]] = state_index if x == (dim -1) else state_index+1 

            '''if your y-coordinate is all the way at the bottom (i.e. y == 2), you stay in the same place -- otherwise you move one down (achieved by adding 3 to your linear state index)'''
            P[state_index][actions['DOWN']] = state_index if y == (dim -1) else state_index + dim 

            ''' if your x-coordinate is all the way at the left (i.e. x == 0), you stay at the same place -- otherwise, you move one to the left (achieved by subtracting 1 from your linear state index)'''
            P[state_index][actions['LEFT']] = state_index if x == 0 else state_index -1 

            ''' Stay in the same place (self explanatory) '''
            P[state_index][actions['STAY']] = state_index
        
        num_states = 9
        B = np.zeros([num_states, num_states, len(actions)])
        for s in range(num_states):
            for a in range(len(actions)):
                ns = int(P[s][a])
                B[ns, s, a] = 1
        
        self.assertTrue(B.shape[0] == 9)

        fig, axes = plt.subplots(2,3, figsize = (15,8))
        a = list(actions.keys())
        count = 0
        for i in range(dim-1):
            for j in range(dim):
                if count >= 5:
                    break 
                g = sns.heatmap(B[:,:,count], cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)
                g.set_title(a[count])
                count +=1 
        fig.delaxes(axes.flatten()[5])
        plt.tight_layout()
    
    def test_gridworld_activeinference(self):
        """
        This unit test runs the a concise version of the code in the `gridworld_tutorial_1.ipynb` tutorial notebook to make sure it works if things are changed
        """

        from pymdp.maths import spm_log_single as log_stable # @NOTE: we use the `spm_log_single` helper function from the `maths` sub-library of pymdp. This is a numerically stable version of np.log()

        state_mapping = {0: (0,0), 1: (1,0), 2: (2,0), 3: (0,1), 4: (1,1), 5:(2,1), 6: (0,2), 7:(1,2), 8:(2,2)}

        A = np.eye(9)
        def plot_beliefs(Qs, title=""):
            #values = Qs.values[:, 0]
            plt.grid(zorder=0)
            plt.bar(range(Qs.shape[0]), Qs, color='r', zorder=3)
            plt.xticks(range(Qs.shape[0]))
            plt.title(title)
            
        labels = [state_mapping[i] for i in range(A.shape[1])]
        def plot_likelihood(A):
            fig = plt.figure(figsize = (6,6))
            ax = sns.heatmap(A, xticklabels = labels, yticklabels = labels, cbar = False)
            plt.title("Likelihood distribution (A)")
            
        def plot_empirical_prior(B):
            fig, axes = plt.subplots(3,2, figsize=(8, 10))
            actions = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'STAY']
            count = 0
            for i in range(3):
                for j in range(2):
                    if count >= 5:
                        break
                        
                    g = sns.heatmap(B[:,:,count], cmap="OrRd", linewidth=2.5, cbar=False, ax=axes[i,j])

                    g.set_title(actions[count])
                    count += 1
            fig.delaxes(axes.flatten()[5])
            plt.tight_layout()
            
        def plot_transition(B):
            fig, axes = plt.subplots(2,3, figsize = (15,8))
            a = list(actions.keys())
            count = 0
            for i in range(dim-1):
                for j in range(dim):
                    if count >= 5:
                        break 
                    g = sns.heatmap(B[:,:,count], cmap = "OrRd", linewidth = 2.5, cbar = False, ax = axes[i,j], xticklabels=labels, yticklabels=labels)
                    g.set_title(a[count])
                    count +=1 
            fig.delaxes(axes.flatten()[5])
            plt.tight_layout()
        
        A = np.eye(9)
        plot_likelihood(A)

        P = {}
        dim = 3
        actions = {'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'STAY':4}

        for state_index, xy_coordinates in state_mapping.items():
            P[state_index] = {a : [] for a in range(len(actions))}
            x, y = xy_coordinates

            '''if your y-coordinate is all the way at the top (i.e. y == 0), you stay in the same place -- otherwise you move one upwards (achieved by subtracting 3 from your linear state index'''
            P[state_index][actions['UP']] = state_index if y == 0 else state_index - dim 

            '''f your x-coordinate is all the way to the right (i.e. x == 2), you stay in the same place -- otherwise you move one to the right (achieved by adding 1 to your linear state index)'''
            P[state_index][actions["RIGHT"]] = state_index if x == (dim -1) else state_index+1 

            '''if your y-coordinate is all the way at the bottom (i.e. y == 2), you stay in the same place -- otherwise you move one down (achieved by adding 3 to your linear state index)'''
            P[state_index][actions['DOWN']] = state_index if y == (dim -1) else state_index + dim 

            ''' if your x-coordinate is all the way at the left (i.e. x == 0), you stay at the same place -- otherwise, you move one to the left (achieved by subtracting 1 from your linear state index)'''
            P[state_index][actions['LEFT']] = state_index if x == 0 else state_index -1 

            ''' Stay in the same place (self explanatory) '''
            P[state_index][actions['STAY']] = state_index

        
        num_states = 9
        B = np.zeros([num_states, num_states, len(actions)])
        for s in range(num_states):
            for a in range(len(actions)):
                ns = int(P[s][a])
                B[ns, s, a] = 1

        plot_transition(B)
        
        class GridWorldEnv():
    
            def __init__(self,A,B):
                self.A = deepcopy(A)
                self.B = deepcopy(B)
                self.state = np.zeros(9)
                self.state[2] = 1
            
            def step(self,a):
                self.state = np.dot(self.B[:,:,a], self.state)
                obs = utils.sample(np.dot(self.A, self.state))
                return obs

            def reset(self):
                self.state =np.zeros(9)
                self.state[2] =1 
                obs = utils.sample(np.dot(self.A, self.state))
                return obs
        
        env = GridWorldEnv(A,B)

        def KL_divergence(q,p):
            return np.sum(q * (log_stable(q) - log_stable(p)))

        def compute_free_energy(q,A, B):
            return np.sum(q * (log_stable(q) - log_stable(A) - log_stable(B)))

        def softmax(x):
            return np.exp(x) / np.sum(np.exp(x))

        def perform_inference(likelihood, prior):
            return softmax(log_stable(likelihood) + log_stable(prior))
        
        Qs = np.ones(9) * 1/9
        plot_beliefs(Qs)

        REWARD_LOCATION = 7
        reward_state = state_mapping[REWARD_LOCATION]

        C = np.zeros(num_states)
        C[REWARD_LOCATION] = 1. 
        plot_beliefs(C)

        def evaluate_policy(policy, Qs, A, B, C):
            # initialize expected free energy at 0
            G = 0

            # loop over policy
            for t in range(len(policy)):

                # get action entailed by the policy at timestep `t`
                u = int(policy[t])

                # work out expected state, given the action
                Qs_pi = B[:,:,u].dot(Qs)

                # work out expected observations, given the action
                Qo_pi = A.dot(Qs_pi)

                # get entropy
                H = - (A * log_stable(A)).sum(axis = 0)

                # get predicted divergence
                # divergence = np.sum(Qo_pi * (log_stable(Qo_pi) - log_stable(C)), axis=0)
                divergence = KL_divergence(Qo_pi, C)
                
                # compute the expected uncertainty or ambiguity 
                uncertainty = H.dot(Qs_pi)

                # increment the expected free energy counter for the policy, using the expected free energy at this timestep
                G += (divergence + uncertainty)

            return -G

        def infer_action(Qs, A, B, C, n_actions, policies):
    
            # initialize the negative expected free energy
            neg_G = np.zeros(len(policies))

            # loop over every possible policy and compute the EFE of each policy
            for i, policy in enumerate(policies):
                neg_G[i] = evaluate_policy(policy, Qs, A, B, C)

            # get distribution over policies
            Q_pi = maths.softmax(neg_G)

            # initialize probabilites of control states (convert from policies to actions)
            Qu = np.zeros(n_actions)

            # sum probabilites of control states or actions 
            for i, policy in enumerate(policies):
                # control state specified by policy
                u = int(policy[0])
                # add probability of policy
                Qu[u] += Q_pi[i]

            # normalize action marginal
            utils.norm_dist(Qu)

            # sample control from action marginal
            u = utils.sample(Qu)

            return u

        # number of time steps
        T = 10

        #n_actions = env.n_control
        n_actions = 5

        # length of policies we consider
        policy_len = 4

        # this function generates all possible combinations of policies
        policies = control.construct_policies([B.shape[0]], [n_actions], policy_len)

        # reset environment
        o = env.reset()

        # loop over time
        for t in range(T):

            # infer which action to take
            a = infer_action(Qs, A, B, C, n_actions, policies)
            
            # perform action in the environment and update the environment
            o = env.step(int(a))
            
            # infer new hidden state (this is the same equation as above but with PyMDP functions)
            likelihood = A[o,:]
            prior = B[:,:,int(a)].dot(Qs)

            Qs = maths.softmax(log_stable(likelihood) + log_stable(prior))
            
            plot_beliefs(Qs, "Beliefs (Qs) at time {}".format(t))

        # self.assertEqual(np.argmax(Qs), REWARD_LOCATION) # @NOTE: This is not always true due to stochastic samplign!!!
        self.assertEqual(Qs.shape[0], B.shape[0])



           
if __name__ == "__main__":
    unittest.main()