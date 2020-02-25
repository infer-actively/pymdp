#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Environment Base Class

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

from inferactively.envs import Env
from inferactively.distributions import Categorical
from inferactively.core import core
import numpy as np

class GridWorldEnv(Env):
    """
    Class definition for a 2-dimension grid world environment with two hidden state factors:
    1) First factor indicates the location of the agent in the world
    2) Second factor indicates the identity of the grid that the agent is currently on (1 of numColors different colors)
    """ 

    def __init__(self, num_grids=[8, 8], num_colors = 3, obs_precis = [1e16, 1e16]):

        """Initialize the two-factor grid world environment
        Parameters
        ----------
        `num_grids` [list :: int] 
            Specifies the size of the grid world in x and y dimensions
        `num_colors` [int]
            Specifies the number of different colors that characterize the second hidden state factor
        """

        self.Ns = [np.prod(num_grids), num_colors]
        self.Nf = len(self.Ns)

        self.A = self.get_likelihood_dist(precisions = obs_precis)

        self.No = [arr.shape[0] for arr in self.A]
        self.Ng = len(self.No)

        self.B = self.get_transition_dist(num_grids)

    def step(self, action):

        if isinstance(self.state, tuple):
            curr_state = np.empty(self.Nf, dtype = object)
            for f in range(self.Nf):
                curr_state[f] = np.eye(self.Ns[f])[self.state[f]]
        elif not isinstance(self.state, (Categorical, np.ndarray)):
            curr_state = self.state

        ps = np.empty(self.Nf, dtype = object)
        for f in range(self.Nf):
            ps[f] = self.B[f][:,:,action[f]].dot(curr_state[f],return_numpy=True).flatten()
        
        ps = Categorical(values = ps)
        # ps = Categorical( values = np.array( [self.B[f][:,:,action[f]].dot(curr_state[f],return_numpy=True) for f in range(self.Nf)], dtype=object))
        next_state = ps.sample()

        self.state = next_state

        return self.state
    
    def get_obs(self):
        
        if isinstance(self.state, tuple):
            curr_state = np.empty(self.Nf, dtype = object)
            for f in range(self.Nf):
                curr_state[f] = np.eye(self.Ns[f])[self.state[f]]
        elif not isinstance(self.state, (Categorical, np.ndarray)):
            curr_state = self.state
        
        po = np.empty(self.Ng, dtype = object)
        for g in range(self.Ng):
            po[g] = self.A[g].dot(curr_state,return_numpy=True).flatten()
        
        po = Categorical(values = po)
        obs = po.sample()

        return obs
        
    def reset(self, state):
        self.state = state
        return self.state

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        return

    def get_likelihood_dist(self, precisions = [1e16, 1e16], return_numpy = False):
        """
        Generate observation likelihoods for the two hidden state factors, also using a vector of precisions
        """

        Ng = self.Nf # number of outcome modalities is equal to the number of hidden state factors

        A = np.empty(Ng, dtype = object)

        grid_obs_lh = core.softmax(np.eye(self.Ns[0]) * precisions[0])
        color_obs_lh = core.softmax(np.eye(self.Ns[1]) * precisions[1])

        A[0] = np.repeat(grid_obs_lh[:,:,np.newaxis],self.Ns[1],axis=2)
        A[1] = np.moveaxis(np.repeat(color_obs_lh[:,:,np.newaxis],self.Ns[0],axis=2), 1, 2)

        if return_numpy:
            return A
        else:
            return Categorical(values = A)    

    def get_transition_dist(self, num_grids, return_numpy = False):
        """
        Programmatically generate 'local action' B matrices that correspond to five different actions:
        1. MOVE UP
        2. MOVE DOWN
        3. MOVE LEFT
        4. MOVE RIGHT
        5. STAY (MOVE TO SAME POSITION AS LAST TIME)
        Parameters
        ----------
        num_grids [list::int]:
            Number of grid tiles in the x direction and y directions
        return_numpy [Bool]:
            Logical flag indicating whether to return a numpy array or a Categorical
        Returns
        -------
        B [numpy object array-of-arrays]:
            Transition likelihoods, with action-conditioned transitions stored along third axis (axis = 2, in numpy lingo)
        """

        B = np.empty(self.Nf, dtype=object)

        x_tiles, y_tiles = num_grids

        # First create B[0]

        Nu = 5
        u = np.zeros((Nu,2)) # used to create the five possible 'local' actions - UP, DOWN, LEFT, RIGHT, STAY
        u[0,1] = 1
        u[1,1] = -1
        u[2,0] = 1
        u[3,0] = -1

        B_0 = np.zeros( (self.Ns[0], self.Ns[0], Nu))
        for x in range(x_tiles):
            for y in range(y_tiles):
                s = np.ravel_multi_index( (np.array([x]), np.array([y])), dims = (x_tiles, y_tiles) )
                for k in range(Nu):
                    grid_indices = ( np.array([x + u[k,0]]).astype(int), np.array([y + u[k,1]]).astype(int) )
                    try:
                        ss = np.ravel_multi_index(grid_indices, dims = (x_tiles, y_tiles) )
                        B_0[ss,s,k] = 1.0
                    except:
                        B_0[s,s,k] = 1.0
        
        B[0] = B_0

        # Now create B[1]
        B[1] = np.eye(self.Ns[1]).reshape(self.Ns[1],self.Ns[1],1)

        if return_numpy:
            return B
        else:
            return Categorical(values = B)

    def __str__(self):
        return "<{} instance>".format(type(self).__name__)
