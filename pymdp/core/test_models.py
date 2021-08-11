
import numpy as np
from pymdp.core import utils, maths

def generate_epistemic_MAB_model():
    '''
    Create the generative model matrices (A, B, C, D) for the 'epistemic multi-armed bandit',
    used in the agent_demo
    ''' 
    
    """
    HIDDEN STATE FACTOR LEVEL MAPPINGS
    """
    HIGH_REW, LOW_REW = 0, 1 
    START, PLAYING, SAMPLING = 0, 1, 2
    num_states = [2, 3]  

    """
    OBSERVATION MODALITY LEVEL MAPPINGS
    """
    HIGH_REW_EVIDENCE, LOW_REW_EVIDENCE, NO_EVIDENCE  = 0, 1, 2
    REWARD, PUN, NEUTRAL = 0, 1, 2
    START_O, PLAY_O, SAMPLE_O = 0, 1, 2
    num_obs = [3, 3, 3]

    """
    CONTROL FACTOR LEVEL MAPPINGS
    """
    NULL_ACTION = 0
    START_ACTION, PLAY_ACTION, SAMPLE_ACTION = 0, 1, 2
    num_controls = [1, 3] 

    A = utils.obj_array_zeros([[o] + num_states for _, o in enumerate(num_obs)])

    """
    MODALITY 0 -- INFORMATION-ABOUT-REWARD-STATE MODALITY
    """
    A[0][NO_EVIDENCE,:,  START] = 1.0
    A[0][NO_EVIDENCE, :, PLAYING] = 1.0 
    A[0][HIGH_REW_EVIDENCE, HIGH_REW, SAMPLING] = 0.8
    A[0][LOW_REW_EVIDENCE, HIGH_REW, SAMPLING] = 0.2
    A[0][HIGH_REW_EVIDENCE, LOW_REW, SAMPLING] = 0.2
    A[0][LOW_REW_EVIDENCE, LOW_REW, SAMPLING] = 0.8

    """
    MODALITY 1 -- REWARD MODALITY
    """
    A[1][NEUTRAL, :, START] = 1.0 
    A[1][NEUTRAL, :, SAMPLING] = 1.0 
    A[1][REWARD, HIGH_REW, PLAYING] = maths.softmax(np.array([1.0, 0])) [0]
    A[1][PUN, HIGH_REW, PLAYING] = maths.softmax(np.array([1.0, 0])) [1]
    A[1][REWARD, LOW_REW, PLAYING] = maths.softmax(np.array([0.0, 1.0]))[0]
    A[1][PUN, LOW_REW, PLAYING] = maths.softmax(np.array([0.0, 1.0]))[1]

    """
    MODALITY 2 -- LOCATION-OBSERVATION MODALITY
    """
    A[2][START_O,:,START] = 1.0
    A[2][PLAY_O,:,PLAYING] = 1.0
    A[2][SAMPLE_O,:,SAMPLING] = 1.0


    control_fac_idx = [1] # this is the controllable control state factor, where there will be a >1-dimensional control state along this factor
    B = utils.obj_array_zeros([[n_s, n_s, num_controls[f]] for f, n_s in enumerate(num_states)])

    """
    FACTOR 0 -- REWARD STATE DYNAMICS
    """

    p_stoch = 0.0

    # we cannot influence factor zero, set up the 'default' stationary dynamics - 
    # one state just maps to itself at the next timestep with very high probability, by default. So this means the reward state can
    # change from one to another with some low probability (p_stoch)

    B[0][HIGH_REW, HIGH_REW, NULL_ACTION] = 1.0 - p_stoch
    B[0][LOW_REW, HIGH_REW, NULL_ACTION] = p_stoch

    B[0][LOW_REW, LOW_REW, NULL_ACTION] = 1.0 - p_stoch
    B[0][HIGH_REW, LOW_REW, NULL_ACTION] = p_stoch
    
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