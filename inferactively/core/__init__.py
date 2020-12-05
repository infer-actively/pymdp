from .maths import (
    spm_dot,
    spm_cross,
    spm_norm,
    spm_wnorm,
    spm_betaln,
    softmax,
    calc_free_energy,
    kl_divergence,
    spm_MDP_G,
)
from .control import (
    update_posterior_policies,
    get_expected_states,
    get_expected_obs,
    calc_expected_utility,
    calc_states_info_gain,
    calc_pA_info_gain,
    calc_pB_info_gain,
    construct_policies,
    sample_action,
)
from .inference import (
    update_posterior_states,
    process_observations,
    process_priors,
    print_inference_methods,
)
from .learning import update_likelihood_dirichlet, update_transition_dirichlet
from .utils import (
    to_numpy,
    is_distribution,
    is_arr_of_arr,
    to_arr_of_arr,
    to_categorical,
    to_dirichlet,
)
from .algos import run_fpi, run_mmp, run_mmp_v2
