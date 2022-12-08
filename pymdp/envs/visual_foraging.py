#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Visual Foraging Environment

__author__: Conor Heins, Alexander Tschantz, Brennan Klein

"""

from pymdp.envs import Env
from pymdp import utils, maths
import numpy as np
from itertools import permutations, product

LOCATION_ID = 0
SCENE_ID = 1

class VisualForagingEnv(Env):
    """ Implementation of the visual foraging environment used for scene construction simulations """

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

scene_names = ["UP_RIGHT", "RIGHT_DOWN", "DOWN_LEFT", "LEFT_UP"] # possible scenes
quadrant_names = ['1','2','3','4']
choice_names = ['choose_UP_RIGHT','choose_RIGHT_DOWN','choose_DOWN_LEFT', 'choose_LEFT_UP'] # possible choices
config_names = list(permutations([1,2,3,4], 2))
all_scenes_all_configs = list(product(scene_names, config_names))

motion_dir = ['null','UP','RIGHT','DOWN','LEFT']
n_states = len(motion_dir)
sampling_states = ['sample', 'break']

class SceneConstruction(Env):
    
    def __init__(self, starting_loc = 'start', scene_name = 'UP_RIGHT', config = "1_2"):

        pos1, pos2 = config.split("_")
        config_tuple = (int(pos1), int(pos2))

        assert scene_name in scene_names, f"{scene_name} is not a possible scene! please choose from {scene_names[0]}, {scene_names[1]}, {scene_names[2]}, or {scene_names[3]}\n"
        assert config_tuple in config_names, f"{config} is not a possible spatial configuration! Please choose an appropriate 2x2 spatial configuration\n"

        self.current_location = starting_loc
        self.scene_name = scene_name
        self.config = config
        self._create_visual_array()

        print(f'Starting location is {self.current_location}, Scene is {self.scene_name}, Configuration is {self.config}\n')

    def step(self,action_label):

        location = self.current_location

        if action_label == 'start': 
          
            new_location = 'start'
            what_obs = 'null'

        elif action_label in quadrant_names:

            what_obs = self.vis_array_flattened[int(action_label)-1]
            new_location = action_label

        elif action_label in choice_names:
            new_location = action_label

            chosen_scene_name = new_location.split('_')[1] + '_' + new_location.split('_')[2]

            if chosen_scene_name== self.scene_name:
                what_obs = 'correct!'
            else:
                what_obs = 'incorrect!'
        
        self.current_location = new_location # store the new grid location

        return what_obs, self.current_location

    def reset(self):
        self.current_location = "start"
        print(f'Re-initialized location to Start location')
        what_obs = 'null'

        return what_obs, self.current_location

    def _create_visual_array(self):
        """ Create scene array """

        vis_array_flattened = np.array(['null', 'null', 'null', 'null'],dtype="<U6")
        dot_dir1, dot_dir2 = self.scene_name.split("_")
        idx1, idx2 = tuple(map(lambda x: int(x) -1, self.config.split("_")))

        vis_array_flattened[idx1] = dot_dir1
        vis_array_flattened[idx2] = dot_dir2

        self.vis_array_flattened = vis_array_flattened
        self.vis_array = vis_array_flattened.reshape(2,2)

class RandomDotMotion(Env):
    """ 
    Implementation of the random-dot motion environment 
    """

    def __init__(self, precision = 1.0, dot_direction = None, sampling_state = None):
        """ Initialize the RDM task using a desired number of directions, the precision (aka coherence) of the motion, 
        a "true dot direction" that generates the observations, and a sampling_state corresponding to how the agent starts (by sampling or not sampling the dot motion)
        """

        if dot_direction is None:
            self._dot_dir = np.random.choice(motion_dir)
        else:
            assert dot_direction in motion_dir, f"{dot_direction} is not a valid motion direction\n"
            self._dot_dir = dot_direction
        
        if sampling_state is None:
            self._action = np.random.choice(sampling_states)
        else:
            self._set_sampling_state(sampling_state)

        self._p = precision
        self.direction_names = motion_dir
        self.sampling_names = sampling_states
        self.n_states = n_states
        self._generate_dot_dist()
        print(f'True motion direction is {self._dot_dir}, motion coherence is {100.0*self.coherence}\n')

    
    def reset(self, dot_direction = None, sampling_state = None):

        if dot_direction is not None:
            self._dot_dir = dot_direction
            self._generate_dot_dist()
        
        if sampling_state is not None:
            self._set_sampling_state(sampling_state)
        
        return self._get_observation()
    
    def step(self, action):
        
        self._set_sampling_state(action)

        return self._get_observation()

    def _generate_dot_dist(self):

        _stateidx = self.direction_names.index(self._dot_dir)
        if self._dot_dir == 'null':
            self.dot_dist = utils.onehot(_stateidx, self.n_states)
        else:
            dot_dist = np.zeros(self.n_states)
            dot_dist[1:] = maths.softmax(self._p * utils.onehot(_stateidx-1, len(self.direction_names)-1))
            self.dot_dist = dot_dist

        return self.dot_dist
    
    def _get_observation(self):

        is_sampling = self._action == 'sample'
        dot_obs = (self.direction_names[utils.sample(self.dot_dist)]) if is_sampling else 'null' # increment the sample by +1 to account for the fact that there's a "null" observation that occupies observation index 0
        action_obs = 'sampling' if is_sampling else 'breaking'

        return dot_obs, action_obs
    
    def _set_sampling_state(self, action):
        assert action in sampling_states, f"{action} is not a valid sampling state\n"
        self._action = action

    @property
    def dot_direction(self):
        return self._dot_dir

    @property
    def num_directions(self):
        return len(self.direction_names)

    @property
    def precision(self):
        return self._p
    
    @property
    def coherence(self):
        return 0. if self._dot_dir == 'null' else self.dot_dist.max()


def create_2x2_array(scene_name, config):
    """
    Helper function for generating array of visual outcomes from the type and configuration
    """

    flattened_scene_array = np.array(['null', 'null', 'null', 'null'],dtype="<U6")
    dot_dir1, dot_dir2 = scene_name.split("_")
    idx1, idx2 = tuple(map(lambda x: int(x) -1, config))

    flattened_scene_array[idx1] = dot_dir1
    flattened_scene_array[idx2] = dot_dir2

    return flattened_scene_array.reshape(2,2), flattened_scene_array

def initialize_scene_construction_GM(T = 6, reward = 2.0, punishment = -4.0, urgency = -4.0):

    loc_names = ['start'] + quadrant_names + choice_names
    what_obs_names = ['null','UP','RIGHT','DOWN','LEFT','correct!','incorrect!']
    where_obs_names = ['start'] + quadrant_names + choice_names
    action_names = ['start'] + quadrant_names + choice_names

    num_states   = [len(all_scenes_all_configs), len(loc_names)]
    num_obs      = [len(what_obs_names), len(where_obs_names)]            # 7 possible visual outcomes (what I'm looking at: "null", "UP", "RIGHT", "DOWN", "LEFT", "CORRECT", "INCORRECT"), 9 possible proprioceptive outcomes (where I'm looking)
    num_controls = [1, len(action_names), 1]

    A = utils.initialize_empty_A(num_obs, num_states)
    B = utils.initialize_empty_B(num_states, num_controls)
    C_shapes = [ [no, T] for no in num_obs]
    C = utils.obj_array_zeros(C_shapes)
    D = utils.obj_array_uniform(num_states)

    # # Create the A array (factorized representation)
    # for scene_id, scene_name in enumerate(scene_names):
    #     for loc_id, loc_name in enumerate(loc_names):
    #         for config_id, config_name in enumerate(config_names):
    #             _, flattened_scene_array = create_2x2_array(scene_name, config_name)
    #             if loc_name == 'start': # at fixation location
    #                 A[0][0, scene_id, loc_id, config_id] = 1.0
    #             elif loc_name in quadrant_names: # fixating one of the quadrants
    #                 A[0][0, scene_id, loc_id, config_id] = 'null' == flattened_scene_array[loc_id-1]
    #                 A[0][1, scene_id, loc_id, config_id] = 'UP' == flattened_scene_array[loc_id-1]
    #                 A[0][2, scene_id, loc_id, config_id] = 'RIGHT' == flattened_scene_array[loc_id-1]
    #                 A[0][3, scene_id, loc_id, config_id] = 'DOWN'  == flattened_scene_array[loc_id-1]
    #                 A[0][4, scene_id, loc_id, config_id] = 'LEFT'  == flattened_scene_array[loc_id-1]
    #             elif loc_name in choice_names: # making a choice

    #                 scene_choice = loc_name.split("_")[1] + "_" + loc_name.split("_")[2]
    #                 A[0][5,scene_id, loc_id, config_id] = scene_choice== scene_name # they get correct feedback if they choose the true scene at play
    #                 A[0][6,scene_id, loc_id, config_id] = scene_choice != scene_name # they get incorrect feedback if they choose anything other than the true scene at play
                
    #             A[1][loc_id, scene_id, loc_id, config_id] = 1.0

    # Create the A array (fully-enumerated parameterization)
    for state_id, scene_and_config_name in enumerate(all_scenes_all_configs):
        scene_name, config_name = scene_and_config_name
        for loc_id, loc_name in enumerate(loc_names):
            _, flattened_scene_array = create_2x2_array(scene_name, config_name)
            if loc_name == 'start': # at fixation location
                A[0][0, state_id, loc_id] = 1.0
            elif loc_name in quadrant_names: # fixating one of the quadrants
                A[0][0, state_id, loc_id] = 'null' == flattened_scene_array[loc_id-1]
                A[0][1, state_id, loc_id] = 'UP' == flattened_scene_array[loc_id-1]
                A[0][2, state_id, loc_id] = 'RIGHT' == flattened_scene_array[loc_id-1]
                A[0][3, state_id, loc_id] = 'DOWN'  == flattened_scene_array[loc_id-1]
                A[0][4, state_id, loc_id] = 'LEFT'  == flattened_scene_array[loc_id-1]
            elif loc_name in choice_names: # making a choice
                scene_choice = loc_name.split("_")[1] + "_" + loc_name.split("_")[2]
                A[0][5,state_id, loc_id] = scene_choice== scene_name # they get correct feedback if they choose the true scene at play
                A[0][6,state_id, loc_id] = scene_choice != scene_name # they get incorrect feedback if they choose anything other than the true scene at play
            
            A[1][loc_id, state_id, loc_id] = 1.0

    control_fac_idx = [1]
    for f, ns in enumerate(num_states):
        if f in control_fac_idx:
            B[f] = utils.construct_controllable_B( [ns], [num_controls[f]] )[0]
        else:
            B[f][:,:,0] = np.eye(ns)

    C[0][5,:] = reward # the agent expects to be right across timesteps
    C[0][6,:] = punishment # the agent expects to not be wrong across timesteps
    C[1][:5,4:] = urgency # make too much exploration costly

    D[1] = utils.onehot(0, num_states[1]) # give agent certain beliefs about starting location

    parameters = {'A': A,
                 'B': B,
                 'C': C,
                 'D': D
                }
    
    mapping = {'scene_names': scene_names,
                'what_obs_names': what_obs_names,
                'where_obs_names': where_obs_names,
                'action_names': action_names
                }

    dimensions = {'num_states': num_states,
                  'num_obs': num_obs,
                  'num_controls': num_controls,
                  }

    
    return parameters, mapping, dimensions

def initialize_RDM_GM(T=16, A_precis = 1.0, break_reward = 0.001):

    sampling_state_names = ['sampling','breaking']
    what_obs_names = ['null','UP','RIGHT','DOWN','LEFT']
    where_obs_names = ['sampling','breaking']
    action_names = ['sample','break']

    n_dir = len(what_obs_names)-1

    num_states = [len(motion_dir), len(sampling_state_names)]
    num_obs = [len(what_obs_names), len(where_obs_names)]
    num_controls = [1, len(action_names)]

    # Initialize A, B, C and D arrays
    A = utils.initialize_empty_A(num_obs=num_obs,num_states=num_states)
    B = utils.initialize_empty_B(num_states=num_states, num_controls=num_controls)
    C = utils.obj_array_zeros(num_obs)
    D = utils.obj_array_uniform(num_states)

    for idx, sampling_state in enumerate(sampling_state_names):
        if sampling_state == 'sampling':
            A[0][0,0,idx] = 1.0
            A[0][1:,1:,idx] = maths.softmax(A_precis * np.eye(n_dir))
            A[1][0,:,idx] = 1.0
        elif sampling_state == 'breaking':
            A[1][idx,:,idx] = 1.0
            A[0][0,:,idx] = 1.0
    
    B[0][:,:,0] = np.eye(num_states[0]) # agent assumes the hidden dot-direction state doesn't change over time
    B[1][0,0,0] = 1.0 # if agent chooses to continnue sampling while already sampling, they keep sampling.
    B[1][1,0,1] = 1.0 # you can move from sampling to breaking
    B[1][1,1,:] = 1.0 # Once you are in the break-state, you cannot "escape" it with either eaction (break-state is an absorbing state or sink in the transition model)

    C[1][1] = break_reward

    start_state_id = sampling_state_names.index('sampling')
    D[1] = utils.onehot(start_state_id, num_states[1]) # prior that agent starts in the sampling state

    parameters = {'A': A,
                 'B': B,
                 'C': C,
                 'D': D
                }
    
    mapping = { 'motion_dir': motion_dir,
                'sampling_state_names': sampling_state_names,
                'what_obs_names': what_obs_names,
                'where_obs_names': where_obs_names,
                'action_names': action_names
                }

    dimensions = {'num_states': num_states,
                  'num_obs': num_obs,
                  'num_controls': num_controls,
                  }

    return parameters, mapping, dimensions
