
import numpy as np
from pymdp import utils, maths

def generate_epistemic_MAB_model():
    '''
    Create the generative model matrices (A, B, C, D) for the 'epistemic multi-armed bandit',
    used in the `agent_demo.py` Python file and the `agent_demo.ipynb` notebook.
    ''' 
    
    num_states = [2, 3]  
    num_obs = [3, 3, 3]
    num_controls = [1, 3] 
    A = utils.obj_array_zeros([[o] + num_states for _, o in enumerate(num_obs)])

    """
    MODALITY 0 -- INFORMATION-ABOUT-REWARD-STATE MODALITY
    """

    A[0][:, :, 0] = np.ones( (num_obs[0], num_states[0]) ) / num_obs[0]
    A[0][:, :, 1] = np.ones( (num_obs[0], num_states[0]) ) / num_obs[0]
    A[0][:, :, 2] = np.array([[0.8, 0.2], [0.0, 0.0], [0.2, 0.8]])

    """
    MODALITY 1 -- REWARD MODALITY
    """

    A[1][2, :, 0] = np.ones(num_states[0])
    A[1][0:2, :, 1] = maths.softmax(np.eye(num_obs[1] - 1)) # bandit statistics (mapping between reward-state (first hidden state factor) and rewards (Good vs Bad))
    A[1][2, :, 2] = np.ones(num_states[0])

    """
    MODALITY 2 -- LOCATION-OBSERVATION MODALITY
    """
    A[2][0,:,0] = 1.0
    A[2][1,:,1] = 1.0
    A[2][2,:,2] = 1.0

    control_fac_idx = [1] # this is the controllable control state factor, where there will be a >1-dimensional control state along this factor
    B = utils.obj_array_zeros([[n_s, n_s, num_controls[f]] for f, n_s in enumerate(num_states)])

    """
    FACTOR 0 -- REWARD STATE DYNAMICS
    """

    p_stoch = 0.0

    # we cannot influence factor zero, set up the 'default' stationary dynamics - 
    # one state just maps to itself at the next timestep with very high probability, by default. So this means the reward state can
    # change from one to another with some low probability (p_stoch)

    B[0][0, 0, 0] = 1.0 - p_stoch
    B[0][1, 0, 0] = p_stoch

    B[0][1, 1, 0] = 1.0 - p_stoch
    B[0][0, 1, 0] = p_stoch
    
    """
    FACTOR 1 -- CONTROLLABLE LOCATION DYNAMICS
    """
    # setup our controllable factor.
    B[1] = utils.construct_controllable_B(num_states, num_controls)[1]

    C = utils.obj_array_zeros(num_obs)
    C[1][0] = 1.0  # make the observation we've a priori named `REWARD` actually desirable, by building a high prior expectation of encountering it 
    C[1][1] = -1.0    # make the observation we've a prior named `PUN` actually aversive,by building a low prior expectation of encountering it

    control_fac_idx = [1]

    return A, B, C, control_fac_idx

def generate_grid_world_transitions(action_labels, num_rows = 3, num_cols = 3):
    """ 
    Wrapper code for creating the controllable transition matrix 
    that an agent can use to navigate in a 2-dimensional grid world
    """

    num_grid_locs = num_rows * num_cols

    transition_matrix = np.zeros( (num_grid_locs, num_grid_locs, len(action_labels)) )

    grid = np.arange(num_grid_locs).reshape(num_rows, num_cols)
    it = np.nditer(grid, flags=["multi_index"])

    loc_list = []
    while not it.finished:
        loc_list.append(it.multi_index)
        it.iternext()

    for action_id, action_label in enumerate(action_labels):

        for curr_state, grid_location in enumerate(loc_list):

            curr_row, curr_col = grid_location
            
            if action_label == "LEFT":
                next_col = curr_col - 1 if curr_col > 0 else curr_col 
                next_row = curr_row
            elif action_label == "DOWN":
                next_row = curr_row + 1 if curr_row < (num_rows-1) else curr_row 
                next_col = curr_col
            elif action_label == "RIGHT":
                next_col = curr_col + 1 if curr_col < (num_cols-1) else curr_col 
                next_row = curr_row
            elif action_label == "UP":
                next_row = curr_row - 1 if curr_row > 0 else curr_row 
                next_col = curr_col
            elif action_label == "STAY":
                next_row, next_col = curr_row, curr_col

            new_location = (next_row, next_col)
            next_state = loc_list.index(new_location)
            transition_matrix[next_state, curr_state, action_id] = 1.0
    
    return transition_matrix
