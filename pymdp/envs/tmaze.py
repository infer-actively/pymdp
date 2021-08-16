#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" T Maze Environment (Factorized)

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

from pymdp.envs import Env
from pymdp import utils, maths
import numpy as np

LOCATION_FACTOR_ID = 0
TRIAL_FACTOR_ID = 1

LOCATION_MODALITY_ID = 0
REWARD_MODALITY_ID = 1
CUE_MODALITY_ID = 2

REWARD_IDX = 1
LOSS_IDX = 2


class TMazeEnv(Env):
    def __init__(self, reward_probs=None):

        if reward_probs is None:
            a = 0.98
            b = 1.0 - a
            self.reward_probs = [a, b]
        else:
            if sum(reward_probs) != 1:
                raise ValueError("Reward probabilities must sum to 1!")
            elif len(reward_probs) != 2:
                raise ValueError("Only two reward conditions currently supported...")
            else:
                self.reward_probs = reward_probs

        self.num_states = [4, 2]
        self.num_locations = self.num_states[LOCATION_FACTOR_ID]
        self.num_controls = [self.num_locations, 1]
        self.num_reward_conditions = self.num_states[TRIAL_FACTOR_ID]
        self.num_cues = self.num_reward_conditions
        self.num_obs = [self.num_locations, self.num_reward_conditions + 1, self.num_cues]
        self.num_factors = len(self.num_states)
        self.num_modalities = len(self.num_obs)

        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()

        self._reward_condition = None
        self._state = None
    
    def reset(self, state=None):
        if state is None:
            loc_state = utils.onehot(0, self.num_locations)
            
            self._reward_condition = np.random.randint(self.num_reward_conditions) # randomly select a reward condition
            reward_condition = utils.onehot(self._reward_condition, self.num_reward_conditions)

            full_state = utils.obj_array(self.num_factors)
            full_state[LOCATION_FACTOR_ID] = loc_state
            full_state[TRIAL_FACTOR_ID] = reward_condition
            self._state = full_state
        else:
            self._state = state
        return self._get_observation()

    def step(self, actions):
        prob_states = utils.obj_array(self.num_factors)
        for factor, state in enumerate(self._state):
            prob_states[factor] = self._transition_dist[factor][:, :, int(actions[factor])].dot(state)
        state = [utils.sample(ps_i) for ps_i in prob_states]
        self._state = self._construct_state(state)
        return self._get_observation()

    def render(self):
        pass

    def sample_action(self):
        return [np.random.randint(self.num_controls[i]) for i in range(self.num_factors)]

    def get_likelihood_dist(self):
        return self._likelihood_dist

    def get_transition_dist(self):
        return self._transition_dist


    def get_rand_likelihood_dist(self):
        pass

    def get_rand_transition_dist(self):
        pass

    def _get_observation(self):

        prob_obs = [maths.spm_dot(A_m, self._state) for A_m in self._likelihood_dist]

        obs = [utils.sample(po_i) for po_i in prob_obs]
        return obs

    def _construct_transition_dist(self):
        B_locs = np.eye(self.num_locations)
        B_locs = B_locs.reshape(self.num_locations, self.num_locations, 1)
        B_locs = np.tile(B_locs, (1, 1, self.num_locations))
        B_locs = B_locs.transpose(1, 2, 0)

        B = utils.obj_array(self.num_factors)

        B[LOCATION_FACTOR_ID] = B_locs
        B[TRIAL_FACTOR_ID] = np.eye(self.num_reward_conditions).reshape(
            self.num_reward_conditions, self.num_reward_conditions, 1
        )
        return B

    def _construct_likelihood_dist(self):

        A = utils.obj_array_zeros([ [obs_dim] + self.num_states for _, obs_dim in enumerate(self.num_obs)] )

        for loc in range(self.num_states[LOCATION_FACTOR_ID]):
            for reward_condition in range(self.num_states[TRIAL_FACTOR_ID]):

                # The case when the agent is in the centre location
                if loc == 0:
                    # When in the centre location, reward observation is always 'no reward'
                    # or the outcome with index 0
                    A[REWARD_MODALITY_ID][0, loc, reward_condition] = 1.0

                    # When in the centre location, cue is totally ambiguous with respect to the reward condition
                    A[CUE_MODALITY_ID][:, loc, reward_condition] = 1.0 / self.num_obs[2]

                # The case when loc == 3, or the cue location ('bottom arm')
                elif loc == 3:

                    # When in the cue location, reward observation is always 'no reward'
                    # or the outcome with index 0
                    A[REWARD_MODALITY_ID][0, loc, reward_condition] = 1.0

                    # When in the cue location, the cue indicates the reward condition umambiguously
                    # signals where the reward is located
                    A[CUE_MODALITY_ID][reward_condition, loc, reward_condition] = 1.0

                # The case when the agent is in one of the (potentially-) rewarding armS
                else:

                    # When location is consistent with reward condition
                    if loc == (reward_condition + 1):
                        # Means highest probability is concentrated over reward outcome
                        high_prob_idx = REWARD_IDX
                        # Lower probability on loss outcome
                        low_prob_idx = LOSS_IDX
                    else:
                        # Means highest probability is concentrated over loss outcome
                        high_prob_idx = LOSS_IDX
                        # Lower probability on reward outcome
                        low_prob_idx = REWARD_IDX

                    reward_probs = self.reward_probs[0]
                    A[REWARD_MODALITY_ID][high_prob_idx, loc, reward_condition] = reward_probs

                    reward_probs = self.reward_probs[1]
                    A[REWARD_MODALITY_ID][low_prob_idx, loc, reward_condition] = reward_probs

                    # Cue is ambiguous when in the reward location
                    A[CUE_MODALITY_ID][:, loc, reward_condition] = 1.0 / self.num_obs[2]

                # The agent always observes its location, regardless of the reward condition
                A[LOCATION_MODALITY_ID][loc, loc, reward_condition] = 1.0

        return A

    def _construct_state(self, state_tuple):

        state = utils.obj_array(self.num_factors)
        for f, ns in enumerate(self.num_states):
            state[f] = utils.onehot(state_tuple[f], ns)

        return state

    @property
    def state(self):
        return self._state

    @property
    def reward_condition(self):
        return self._reward_condition


class TMazeEnvNullOutcome(Env):
    def __init__(self, reward_probs=None):

        if reward_probs is None:
            a = 0.98
            b = 1.0 - a
            self.reward_probs = [a, b]
        else:
            if sum(reward_probs) != 1:
                raise ValueError("Reward probabilities must sum to 1!")
            elif len(reward_probs) != 2:
                raise ValueError("Only two reward conditions currently supported...")
            else:
                self.reward_probs = reward_probs

        self.num_states = [4, 2]
        self.num_locations = self.num_states[LOCATION_FACTOR_ID]
        self.num_controls = [self.num_locations, 1]
        self.num_reward_conditions = self.num_states[TRIAL_FACTOR_ID]
        self.num_cues = self.num_reward_conditions
        self.num_obs = [self.num_locations, self.num_reward_conditions + 1, self.num_cues + 1]
        self.num_factors = len(self.num_states)
        self.num_modalities = len(self.num_obs)

        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()

        self._reward_condition = None
        self._state = None

    def reset(self, state=None):
        if state is None:
            loc_state = utils.onehot(0, self.num_locations)
            
            self._reward_condition = np.random.randint(self.num_reward_conditions) # randomly select a reward condition
            reward_condition = utils.onehot(self._reward_condition, self.num_reward_conditions)

            full_state = utils.obj_array(self.num_factors)
            full_state[LOCATION_FACTOR_ID] = loc_state
            full_state[TRIAL_FACTOR_ID] = reward_condition
            self._state = full_state
        else:
            self._state = state
        return self._get_observation()

    def step(self, actions):
        prob_states = utils.obj_array(self.num_factors)
        for factor, state in enumerate(self._state):
            prob_states[factor] = self._transition_dist[factor][:, :, int(actions[factor])].dot(state)
        state = [utils.sample(ps_i) for ps_i in prob_states]
        self._state = self._construct_state(state)
        return self._get_observation()


    def sample_action(self):
        return [np.random.randint(self.num_controls[i]) for i in range(self.num_factors)]

    def get_likelihood_dist(self):
        return self._likelihood_dist.copy()

    def get_transition_dist(self):
        return self._transition_dist.copy()

    def _get_observation(self):

        prob_obs = [maths.spm_dot(A_m, self._state) for A_m in self._likelihood_dist]

        obs = [utils.sample(po_i) for po_i in prob_obs]
        return obs

    def _construct_transition_dist(self):
        B_locs = np.eye(self.num_locations)
        B_locs = B_locs.reshape(self.num_locations, self.num_locations, 1)
        B_locs = np.tile(B_locs, (1, 1, self.num_locations))
        B_locs = B_locs.transpose(1, 2, 0)

        B = utils.obj_array(self.num_factors)

        B[LOCATION_FACTOR_ID] = B_locs
        B[TRIAL_FACTOR_ID] = np.eye(self.num_reward_conditions).reshape(
            self.num_reward_conditions, self.num_reward_conditions, 1
        )
        return B

    def _construct_likelihood_dist(self):

        A = utils.obj_array_zeros([ [obs_dim] + self.num_states for _, obs_dim in enumerate(self.num_obs)] )
        
        for loc in range(self.num_states[LOCATION_FACTOR_ID]):
            for reward_condition in range(self.num_states[TRIAL_FACTOR_ID]):

                if loc == 0:  # the case when the agent is in the centre location
                    # When in the centre location, reward observation is always 'no reward', or the outcome with index 0
                    A[REWARD_MODALITY_ID][0, loc, reward_condition] = 1.0

                    # When in the center location, cue observation is always 'no cue', or the outcome with index 0
                    A[CUE_MODALITY_ID][0, loc, reward_condition] = 1.0

                # The case when loc == 3, or the cue location ('bottom arm')
                elif loc == 3:

                    # When in the cue location, reward observation is always 'no reward', or the outcome with index 0
                    A[REWARD_MODALITY_ID][0, loc, reward_condition] = 1.0

                    # When in the cue location, the cue indicates the reward condition umambiguously
                    # signals where the reward is located
                    A[CUE_MODALITY_ID][reward_condition + 1, loc, reward_condition] = 1.0

                # The case when the agent is in one of the (potentially-) rewarding arms
                else:

                    # When location is consistent with reward condition
                    if loc == (reward_condition + 1):
                        # Means highest probability is concentrated over reward outcome
                        high_prob_idx = REWARD_IDX
                        # Lower probability on loss outcome
                        low_prob_idx = LOSS_IDX  #
                    else:
                        # Means highest probability is concentrated over loss outcome
                        high_prob_idx = LOSS_IDX
                        # Lower probability on reward outcome
                        low_prob_idx = REWARD_IDX

                    reward_probs = self.reward_probs[0]
                    A[REWARD_MODALITY_ID][high_prob_idx, loc, reward_condition] = reward_probs
                    reward_probs = self.reward_probs[1]
                    A[REWARD_MODALITY_ID][low_prob_idx, loc, reward_condition] = reward_probs

                    # When in the one of the rewarding arms, cue observation is always 'no cue', or the outcome with index 0
                    A[CUE_MODALITY_ID][0, loc, reward_condition] = 1.0

                # The agent always observes its location, regardless of the reward condition
                A[LOCATION_MODALITY_ID][loc, loc, reward_condition] = 1.0

        return A

    def _construct_state(self, state_tuple):

        state = utils.obj_array(self.num_factors)

        for f, ns in enumerate(self.num_states):
            state[f] = utils.onehot(state_tuple[f], ns)
            
        return state

    @property
    def state(self):
        return self._state

    @property
    def reward_condition(self):
        return self._reward_condition
