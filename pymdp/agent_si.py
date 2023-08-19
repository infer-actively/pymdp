#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:56:11 2023

@author: aswinpaul (aswin.paul@monash.edu)
"""

# This is needed since pymdp as a module is in a different folder than this demo
import os
import sys
from pathlib import Path

path = Path(os.getcwd())
module_path = str(path.parent.parent) + '/'
sys.path.append(module_path)

# importing the existing classical AI agent in pymdp to reuse inference and learning
from pymdp.agent import Agent
from pymdp.utils import random_A_matrix, random_B_matrix, obj_array_zeros, onehot
from pymdp.maths import softmax, softmax_obj_arr, kl_div, entropy

# numpy
import numpy as np
np.random.seed(2022)


class si_agent(Agent):
    """
    # Necessary parameters for SI agent

    # num_states
    # num_obs
    # num_controls

    # Optional parameters
    # planning_horizon (default value 1)
    # A = prior for likelihood A (same structure as pymdp.utils.random_A_matrix(num_obs, num_states))
    # B = prior for transisiton matrix B (same structure as pymdp.utils.random_B_matrix(num_obs, num_states))
    # C = prior for preference dist. C (same structure as pymdp.utils.obj_array_zeros(num_obs))
    # D = 0 prior of hidden-state
    # action_precision (precision for softmax in taking decisions) default value: 1
    # planning_precision (precision for softmax during tree search) default value: 1
    # search_threshold = 1/16 parameter pruning tree search in SI tree-search (default value 1/16)

    # Useful combination functions 
    # agent.step([obs_list], learning = False): 
    Combines Inference, planning, learning, and decision-making
    Generative model will be learned and updated over time if learning = True
    """
    def __init__(self, num_states, num_obs, num_controls,
                 planning_horizon = 1, 
                 A = None, B = None, C = None, D = None, 
                 action_precision = 1,
                 planning_precision = 1, 
                 search_threshold = 1/16):
        
        # Initialising the exisiting pymdp agent for Inference, and Learning
        super().__init__(A = A, B = B, C = C, D = D, 
                         pA = A, pB = B, pD = D,
                        gamma = planning_precision, 
                        alpha = action_precision)
        
        # Parameters for sophisticated inference
        self.N = planning_horizon
        self.tau = 0
        # Tree-pruning parameter in sophisticated inference
        self.tree_threshold = search_threshold
        
    #Sophisticated inference functions for planning and decision making. Author: Aswin Paul
    
    # Melting hidden state factors as single hidden state modality to use locally for planning
    def melting_factors_for_planning(self):
        
        self.EPS_VAL = 1e-16
        self.numS = 1
        self.numA = 1
        for i in self.num_states:
            self.numS *= i
        for i in self.num_controls:
            self.numA *= i

        self.new_num_states = [self.numS]
        self.new_num_factors = len(self.num_states)
        self.new_num_controls = [self.numA]
        self.new_num_obs = self.num_obs
        
        self.A_new = random_A_matrix(self.new_num_obs, self.new_num_states)
        if self.A is not None:
            for i in range(len(self.num_obs)):
                self.A_new[i] = self.A[i].reshape(self.new_num_obs[i], self.numS)

        self.B_new = random_B_matrix(self.new_num_states, self.new_num_controls)
        if self.B is not None:
            bb = 1
            for i in range(len(self.num_states)):
                bb = np.kron(bb, self.B[i])
            self.B_new[0] = bb
            
        if self.C is not None:
            self.C_new = softmax_obj_arr(self.C)
        else:
            self.C_new = obj_array_zeros(self.num_obs)
            for idx in range(len(self.num_obs)):
                self.C_new[idx] += (1/self.num_obs[idx])
                
        if self.D is not None:
            self.D_new = self.D
        else:
            self.D_new = obj_array_zeros(self.new_num_states)
            for idx in range(len(self.num_states)):
                self.D_new[idx] += 1 / self.num_states[idx]
                
    # Planning with forward tree search (sophisticated inference)
    def plan_tree_search(self, modalities = False):
        
        self.melting_factors_for_planning()
        
        self.G = np.zeros((self.numA))
        self.Q_actions = np.zeros((self.numA))
        
        self.qs_new = obj_array_zeros(self.new_num_states)
        if self.qs is not None:
            q = 1
            for i in range(len(self.num_states)):
                q = np.kron(q, self.qs[i])
            self.qs_new[0] = q

        #print("Planning")

        if(modalities == False):
            moda = list(range(self.num_modalities))
        else:
            moda = modalities

        for mod in moda:
            N = 0
            self.countt = 0
            if(self.N < 1):
                print("No planning, agent will take equi-probable random actions")
            else:
                self.G = self.forward_search(mod, N, pre = self.qs_new[0])
                
        self.q_pi = softmax(-1*self.alpha*self.G)

    def forward_search(self, mod, N, pre):
        self.countt += 1
        N += 1
        # Tree search
        Q_po = np.zeros((self.A_new[mod].shape[0], self.numA))
        G = np.zeros((self.numA))
        post_l = []
        for j in range(self.numA):
            post = np.matmul(self.B_new[0][:,:,j], pre)
            post_l.append(post)
            Q_po[:,j] = self.A_new[mod].dot(post)
            val = kl_div(Q_po[:,j],self.C_new[mod]) + np.dot(post, entropy(self.A_new[mod]))
            G[j] += val
        # Proxy action selection distribution
        self.Q_actions[:] = softmax(-1*self.gamma*G[:])
        # Further tree search till planning horizon
        if(N < self.N):
            # Tree search pruning over actions
            for j in list(np.where(self.Q_actions[:] >= self.tree_threshold)[0]):
                # Tree search pruning over states
                for i in list(np.where(post_l[j] >= self.tree_threshold)[0]):
                    # Proxy next state belief
                    i_state = onehot(i, self.numS)
                    # Recursive calling of this function for tree-search
                    G_next = self.forward_search(mod, N, i_state)
                    # Proxy action selection distribution
                    self.Q_actions[:] = softmax(-1*self.gamma*G_next[:])
                    # Expectation of EFE of next state over states and actions
                    a = np.multiply(self.Q_actions[:],G_next[:])
                    b = np.reshape(a, (1,self.numA))
                    state_next = np.reshape(post_l[j], (self.numS,1))
                    val = np.sum(np.matmul(state_next,b))
                    # Adding normalised next_stage EFE to exisiting EFE
                    G[j] += val
        return G
    
    #Step function combining exisisting functions for agent-environment loop
    def step(self, obs_list, learning = True):
        """
        Agent step combines the following agent functions:
        Combines Inference, Planning, Learning, and decision-making.
        This function represents the agent-environment loop in behaviour where an "environment" feeds observations
        to an "Agent", then the "Agent" responds with actions to control the "environment".
        Usage: agent.step([obs_list])
        Returns: Action(s) from agent to environment
        """
        if(self.tau == 0):
            # Inference
            self.infer_states(obs_list)
            self.qs_prev = np.copy(self.qs)
            # Planning
            self.plan_tree_search()
            
            # Decision making
            self.sample_action()
            self.tau += 1
            
            # Learning D
            if(learning == True):
                self.update_D(self.qs)

        else:
            # Inference
            self.qs_prev = np.copy(self.qs)
            self.infer_states(obs_list)

            # Learning model parameters
            if(learning == True):
                # Updating b
                self.update_B(self.qs_prev)
                # Updating a
                self.update_A(obs_list)
            
            # Planning
            self.plan_tree_search()
            
            # Decision making
            self.sample_action()
            self.tau += 1

        return(self.action)