#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Visual Foraging Environment

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

from pymdp.envs import Env
import numpy as np

LOCATION_ID = 0
SCENE_ID = 1


class VisualForagingEnv(Env):
    def __init__(self, scenes=None, n_features=2):
        if scenes is None:
            self.scenes = self._construct_default_scenes()
        else:
            self.scenes = scenes

        self.n_scenes = len(self.scenes)
        self.n_features = n_features + 1
        self.n_states = [np.prod(self.scenes[0].shape) + 1, self.scenes.shape[0]]
        self.n_locations = self.n_states[LOCATION_ID]
        self.n_control = [self.n_locations, 1]
        self.n_observations = [self.n_locations, self.n_features]
        self.n_factors = len(self.n_states)
        self.n_modalities = len(self.n_observations)

        self._transition_dist = self._construct_transition_dist()
        self._likelihood_dist = self._construct_likelihood_dist()
        self._true_scene = None
        self._state = None

    def reset(self, state=None):
        if state is None:
            loc_state = np.zeros(self.n_locations)
            loc_state[0] = 1.0
            scene_state = np.zeros(self.n_scenes)
            self._true_scene = np.random.randint(self.n_scenes)
            scene_state[self._true_scene] = 1.0
            full_state = np.empty(self.n_factors, dtype=object)
            full_state[LOCATION_ID] = loc_state
            full_state[SCENE_ID] = scene_state
            self._state = Categorical(values=full_state)
        else:
            self._state = Categorical(values=state)
        return self._get_observation()

    def step(self, actions):
        prob_states = np.empty(self.n_factors, dtype=object)
        for f in range(self.n_factors):
            prob_states[f] = (
                self._transition_dist[f][:, :, actions[f]]
                .dot(self._state[f], return_numpy=True)
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
            [
                np.ones(self.n_states[f]) / self.n_states[f]
                for f in range(self.n_factors)
            ]
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
        B[LOCATION_ID] = B_locs
        B[SCENE_ID] = np.eye(self.n_scenes).reshape(self.n_scenes, self.n_scenes, 1)
        return Categorical(values=B)

    def _construct_likelihood_dist(self):
        A = np.empty(self.n_modalities, dtype=object)
        for g in range(self.n_modalities):
            A[g] = np.zeros([self.n_observations[g]] + self.n_states)

        for loc in range(self.n_states[LOCATION_ID]):
            for scene_id in range(self.n_states[SCENE_ID]):
                scene = self.scenes[scene_id]
                feat_loc_ids = np.ravel_multi_index(np.where(scene), scene.shape)
                if loc in feat_loc_ids + 1:
                    feat_ids = np.unravel_index(
                        feat_loc_ids[loc == (feat_loc_ids + 1)], scene.shape
                    )
                    feats = scene[feat_ids]
                    A[SCENE_ID][int(feats), loc, scene_id] = 1.0
                else:
                    A[SCENE_ID][0, loc, scene_id] = 1.0

                A[LOCATION_ID][loc, loc, scene_id] = 1.0
        return Categorical(values=A)

    def _construct_default_scenes(self):
        scene_one = [[2, 2], [2, 2]]
        scene_two = [[1, 1], [1, 1]]
        scenes = np.array([scene_one, scene_two])
        return scenes

    def _construct_state(self, state_tuple):
        state = np.empty(self.n_factors, dtype=object)
        for f in range(self.n_factors):
            state[f] = np.eye(self.n_states[f])[state_tuple[f]]
        return Categorical(values=state)

    @property
    def state(self):
        return self._state

    @property
    def true_scene(self):
        return self._true_scene
