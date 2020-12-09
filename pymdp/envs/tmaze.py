#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" T Maze Environment (Factorized)

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

from pymdp.envs import Env
from pymdp.distributions import Categorical
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

        self.n_states = [4, 2]
        self.n_locations = self.n_states[LOCATION_FACTOR_ID]
        self.n_control = [self.n_locations, 1]
        self.n_reward_conditions = self.n_states[TRIAL_FACTOR_ID]
        self.n_cues = self.n_reward_conditions
        self.n_observations = [self.n_locations, self.n_reward_conditions + 1, self.n_cues]
        self.n_factors = len(self.n_states)
        self.n_modalities = len(self.n_observations)

        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()

        self._reward_condition = None
        self._state = None

    def reset(self, state=None):
        if state is None:
            loc_state = np.zeros(self.n_locations)
            loc_state[0] = 1.0
            reward_condition = np.zeros(self.n_reward_conditions)
            self._reward_condition = np.random.randint(self.n_reward_conditions)
            reward_condition[self._reward_condition] = 1.0
            full_state = np.empty(self.n_factors, dtype=object)
            full_state[LOCATION_FACTOR_ID] = loc_state
            full_state[TRIAL_FACTOR_ID] = reward_condition
            self._state = Categorical(values=full_state)
        else:
            self._state = Categorical(values=state)
        return self._get_observation()

    def step(self, actions):
        prob_states = np.empty(self.n_factors, dtype=object)
        for factor, state in enumerate(self._state):
            prob_states[factor] = (
                self._transition_dist[factor][:, :, actions[factor]]
                .dot(state, return_numpy=True)
                .flatten()
            )
        state = Categorical(values=prob_states).sample()
        self._state = self._construct_state(state)
        return self._get_observation()

    def render(self):
        pass

    def sample_action(self):
        return [np.random.randint(self.n_control[i]) for i in range(self.n_factors)]

    def get_likelihood_dist(self):
        return self._likelihood_dist.copy()

    def get_transition_dist(self):
        return self._transition_dist.copy()

    def get_uniform_posterior(self):
        values = np.array(
            [np.ones(self.n_states[f]) / self.n_states[f] for f in range(self.n_factors)]
        )
        return Categorical(values=values)

    def get_rand_likelihood_dist(self):
        pass

    def get_rand_transition_dist(self):
        pass

    def _get_observation(self):

        prob_obs = self._likelihood_dist.dot(self._state)

        return prob_obs.sample()

    def _construct_transition_dist(self):
        B_locs = np.eye(self.n_locations)
        B_locs = B_locs.reshape(self.n_locations, self.n_locations, 1)
        B_locs = np.tile(B_locs, (1, 1, self.n_locations))
        B_locs = B_locs.transpose(1, 2, 0)

        B = np.empty(self.n_factors, dtype=object)
        B[LOCATION_FACTOR_ID] = B_locs
        B[TRIAL_FACTOR_ID] = np.eye(self.n_reward_conditions).reshape(
            self.n_reward_conditions, self.n_reward_conditions, 1
        )
        return Categorical(values=B)

    def _construct_likelihood_dist(self):
        A = np.empty(self.n_modalities, dtype=object)
        for modality in range(self.n_modalities):
            A[modality] = np.zeros([self.n_observations[modality]] + self.n_states)

        for loc in range(self.n_states[LOCATION_FACTOR_ID]):
            for reward_condition in range(self.n_states[TRIAL_FACTOR_ID]):

                # The case when the agent is in the centre location
                if loc == 0:
                    # When in the centre location, reward observation is always 'no reward'
                    # or the outcome with index 0
                    A[REWARD_MODALITY_ID][0, loc, reward_condition] = 1.0

                    # When in the centre location, cue is totally ambiguous with respect to the reward condition
                    A[CUE_MODALITY_ID][:, loc, reward_condition] = 1.0 / self.n_observations[2]

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
                    A[CUE_MODALITY_ID][:, loc, reward_condition] = 1.0 / self.n_observations[2]

                # The agent always observes its location, regardless of the reward condition
                A[LOCATION_MODALITY_ID][loc, loc, reward_condition] = 1.0

        return Categorical(values=A)

    def _construct_state(self, state_tuple):
        state = np.empty(self.n_factors, dtype=object)
        for f in range(self.n_factors):
            state[f] = np.eye(self.n_states[f])[state_tuple[f]]
        return Categorical(values=state)

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

        self.n_states = [4, 2]
        self.n_locations = self.n_states[LOCATION_FACTOR_ID]
        self.n_control = [self.n_locations, 1]
        self.n_reward_conditions = self.n_states[TRIAL_FACTOR_ID]
        self.n_cues = self.n_reward_conditions
        self.n_observations = [self.n_locations, self.n_reward_conditions + 1, self.n_cues + 1]
        self.n_factors = len(self.n_states)
        self.n_modalities = len(self.n_observations)

        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()

        self._reward_condition = None
        self._state = None

    def reset(self, state=None):
        if state is None:
            loc_state = np.zeros(self.n_locations)
            loc_state[0] = 1.0
            reward_condition = np.zeros(self.n_reward_conditions)
            self._reward_condition = np.random.randint(self.n_reward_conditions)
            reward_condition[self._reward_condition] = 1.0
            full_state = np.empty(self.n_factors, dtype=object)
            full_state[LOCATION_FACTOR_ID] = loc_state
            full_state[TRIAL_FACTOR_ID] = reward_condition
            self._state = Categorical(values=full_state)
        else:
            self._state = Categorical(values=state)
        return self._get_observation()

    def step(self, actions):
        prob_states = np.empty(self.n_factors, dtype=object)
        for factor, state in enumerate(self._state):
            prob_states[factor] = (
                self._transition_dist[factor][:, :, actions[factor]]
                .dot(state, return_numpy=True)
                .flatten()
            )
        state = Categorical(values=prob_states).sample()
        self._state = self._construct_state(state)
        return self._get_observation()

    def render(self):
        pass

    def sample_action(self):
        return [np.random.randint(self.n_control[i]) for i in range(self.n_factors)]

    def get_likelihood_dist(self):
        return self._likelihood_dist.copy()

    def get_transition_dist(self):
        return self._transition_dist.copy()

    def get_uniform_posterior(self):
        values = np.array(
            [np.ones(self.n_states[f]) / self.n_states[f] for f in range(self.n_factors)]
        )
        return Categorical(values=values)

    def get_rand_likelihood_dist(self):
        pass

    def get_rand_transition_dist(self):
        pass

    def _get_observation(self):

        prob_obs = self._likelihood_dist.dot(self._state)

        return prob_obs.sample()

    def _construct_transition_dist(self):
        B_locs = np.eye(self.n_locations)
        B_locs = B_locs.reshape(self.n_locations, self.n_locations, 1)
        B_locs = np.tile(B_locs, (1, 1, self.n_locations))
        B_locs = B_locs.transpose(1, 2, 0)

        B = np.empty(self.n_factors, dtype=object)
        B[LOCATION_FACTOR_ID] = B_locs
        B[TRIAL_FACTOR_ID] = np.eye(self.n_reward_conditions).reshape(
            self.n_reward_conditions, self.n_reward_conditions, 1
        )
        return Categorical(values=B)

    def _construct_likelihood_dist(self):
        A = np.empty(self.n_modalities, dtype=object)
        for modality in range(self.n_modalities):
            A[modality] = np.zeros([self.n_observations[modality]] + self.n_states)

        for loc in range(self.n_states[LOCATION_FACTOR_ID]):
            for reward_condition in range(self.n_states[TRIAL_FACTOR_ID]):

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

        return Categorical(values=A)

    def _construct_state(self, state_tuple):
        state = np.empty(self.n_factors, dtype=object)
        for f in range(self.n_factors):
            state[f] = np.eye(self.n_states[f])[state_tuple[f]]
        return Categorical(values=state)

    @property
    def state(self):
        return self._state

    @property
    def reward_condition(self):
        return self._reward_condition
