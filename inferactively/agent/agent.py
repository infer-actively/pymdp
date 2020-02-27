#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" inferactively Agent Class

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

from inferactively.distributions import Categorical, Dirichlet
from inferactively.core import core, inference
import numpy as np

class Agent(object):

    def __init__(self, **kwargs):

        if 'policy_len' in kwargs:
            self.policy_len = kwargs['policy_len']
        else:
            self.policy_len = 1
        
        if 'n_observations' in kwargs:
            self.n_observations =  kwargs['n_observations']
        else:
            self.n_observations = [5]
        
        if 'n_modalities' in kwargs:
            self.n_modalities = kwargs['n_modalities']
            if self.n_modalities != len(self.n_observations):
                raise ValueError("n_modalities must equal len(n_observations)")
        else:
            self.n_modalities = len(self.n_observations)
        
        if 'n_states' in kwargs:
            self.n_states = kwargs['n_states']
        else:
            self.n_states = [5]

        if 'n_factors' in kwargs:
            self.n_factors = kwargs['n_factors']
            if self.n_factors != len(self.n_states):
                raise ValueError("n_factors must equal len(n_states)")
        else:
            self.n_factors = len(self.n_states)

        if 'control_fac_idx' in kwargs:
            self.control_fac_idx = kwargs['control_fac_idx']
            if max(self.control_fac_idx) > (self.n_factors-1):
                raise ValueError('Desired index of controllable factor exceeds of range of hidden state factors')
        else:
            self.control_fac_idx = list(range(self.n_factors))

        if 'n_controls' in kwargs:
            self.n_controls = kwargs['n_controls']
            if len(self.n_controls) != self.n_factors:
                raise ValueError("n_controls must equal n_factors")
        else:
            self.n_controls, _ = self._construct_n_controls()
        
        if 'possible_policies' in kwargs:
            self.possible_policies = kwargs['possible_policies']
        else:
            _, self.possible_policies = self._construct_n_controls()
        
        if 'A' in kwargs:
            if isinstance(kwargs['A'],Categorical):
                self.A = kwargs['A']
            else:
                self.A = Categorical(values = kwargs['A'])   
            if self.A.IS_AOA:
                self.n_modalities = self.A.shape[0]
                self.n_observations = [self.A[g].shape[0] for g in range(self.n_modalities)]
            else:
                self.n_modalities = 1
                self.n_observations = self.A.shape[0]
        else:
            self.A = self._construct_A_likelihood()
        
        if 'B' in kwargs:
            if isinstance(kwargs['B'], Categorical):
                self.B = kwargs['B']
            else:
                self.B = Categorical(values = kwargs['B'])
            if self.B.IS_AOA:
                self.n_factors = self.B.shape[0]
                self.n_states = [self.B[f].shape[0] for f in range(self.n_factors)]
            else:
                self.n_factors = 1
                self.n_states = [self.B.shape[0]]
            if 'control_fac_idx' not in kwargs:
                self.control_fac_idx = list(range(self.n_factors))
            self.n_controls, self.possible_policies = self._construct_n_controls()
        else:
            self.B = self._construct_B_likelihood()

        if 'C' in kwargs:
            self.C = kwargs['C']
        else:
            self.C = self._construct_C_prior()

        if 'D' in kwargs:
            if isinstance(kwargs['D'], Categorical):
                self.D = kwargs['D']
            else:
                self.D = Categorical(values = kwargs['D'])
        else:
            self.D = self._construct_D_prior()
        
        if 'MP_alg' in kwargs:
            self.MP_alg = kwargs['MP_alg']
        else:
            self.MP_alg = 'FPI'

        if 'MP_params' in kwargs:
            self.MP_params = kwargs['MP_params']
        else:
            self.MP_params = self._get_default_params()

        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
        else:
            self.gamma = 16.0
        
        if 'sampling_type' in kwargs:
            self.sampling_type = kwargs['sampling_type']
        else:
            self.sampling_type = 'marginal_action'

        self.qx = self.D
        self.selectedAction = None

        self.pA = None
        self.pB = None

    def _construct_n_controls(self):
            
        n_controls, possible_policies = inference.constructNu(self.n_states, self.n_factors, self.control_fac_idx, self.policy_len)

        return n_controls, possible_policies

    def _construct_A_likelihood(self):
        
        if self.n_modalities == 1:
            A = Categorical(values = np.random.rand(*(self.n_observations + self.n_states)))
        else:
            A = np.empty(self.n_modalities, dtype=object)
            for g, No in enumerate(self.n_observations):
                A[g] = np.random.rand(*([No] + self.n_states))
            
            A = Categorical(values = A)
        A.normalize()
        return A

    def _construct_B_likelihood(self):

        if self.n_factors == 1:
            B = np.eye(*self.n_states)[:,:,np.newaxis]
            if 0 in self.control_fac_idx:
                B = np.tile(B, (1,1,self.n_controls[0]))
                B = B.transpose(1,2,0)
        else:
            B = np.empty(self.n_factors, dtype=object)
            
            for f, Ns in enumerate(self.n_states):
                B_basic = np.eye(Ns)[:,:,np.newaxis]
                if f in self.control_fac_idx:
                    B[f] = np.tile(B_basic, (1, 1, self.n_controls[f]))
                    B[f] = B[f].transpose(1, 2, 0)
                else:
                    B[f] = B_basic

        B = Categorical(values = B)
        B.normalize()
        return B
    
    def _construct_C_prior(self):

        if self.n_modalities == 1:
            C = np.zeros(*self.n_observations)
        else:
            C = np.array( [np.zeros(No) for No in self.n_observations] )

        return C
        
    def _construct_D_prior(self):

        if self.n_factors == 1:
            D = Categorical(values = np.ones(*self.n_states))
        else:
            D = Categorical(values = np.array( [np.ones(Ns) for Ns in self.n_states] ))
        D.normalize()

        return D

    def reset(self, init_qx=None):

        if init_qx is None:
            self.qx = self._construct_D_prior()
        else:
            if isinstance(init_qx, Categorical):
                self.qx = init_qx
            else:
                self.qx = Categorical(values=init_qx)

        return self.qx
    
    def infer_states(self, observation):

        if not hasattr(self,'qx'):
            self.reset()

        if self.MP_alg is "FPI":
            if self.selectedAction is not None:
                empirical_prior = inference.get_expected_states(self.qx, self.B.log(), self.selectedAction)
            else:
                empirical_prior = self.D.log()
        else:
            if self.selectedAction is not None:
                empirical_prior = inference.get_expected_states(self.qx, self.B.log(), self.selectedAction)
            else:
                empirical_prior = self.D

        qx = inference.update_posterior_states(self.A, observation, empirical_prior, return_numpy = False, 
                                                    method=self.MP_alg, **self.MP_params)
        self.qx = qx

        return qx
    
    def infer_policies(self):
        
        q_pi, efe = inference.update_posterior_policies(self.qx, self.A, self.B, self.C, self.possible_policies,
                                                    self.pA, self.pB, self.gamma, return_numpy=False)
        self.efe = efe
        self.q_pi = q_pi
        return q_pi, efe

    def sample_action(self):

        selectedAction = inference.sample_action(self.q_pi,self.possible_policies,self.n_controls,self.sampling_type)

        self.selectedAction = selectedAction
        return selectedAction

    def _get_default_params(self):

        method = self.MP_alg

        if method == 'FPI':
            default_params = {'dF':1.0, 'dF_tol': 0.001}
        if method == "VMP":
            raise NotImplementedError("VMP is not implemented")
        if method == "MMP":
            raise NotImplementedError("MMP is not implemented")
        if method == "BP":
            raise NotImplementedError("BP is not implemented")
        if method == "EP":
            raise NotImplementedError("EP is not implemented")
        if method == "CV":
            raise NotImplementedError("CV is not implemented")

        return default_params


# %%