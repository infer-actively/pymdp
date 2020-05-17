# %%
import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from inferactively.distributions import Categorical, Dirichlet
from inferactively import core

# %%
""" 
@TODO :

1 SINGLE FACTOR 
- 1a Single factor, single timestep
    -- get expected states X
    1a(i)Single Modality
        -- get expected observations X
        --update_posterior_policies (just utility + state info gain) X
        --update_posterior_policies (now with param info gain) X
    1a(ii)Multiple Modality
        -- get expected observations X
        --update_posterior_policies (just utility + state info gain) X
        --update_posterior_policies (now with param info gain) X
- 1b Single factor, multiple timestep
    -- get expected states X
    1b(i) Single Modality
        -- get expected observations X
        --update_posterior_policies (just utility + state info gain) X
        --update_posterior_policies (now with param info gain) X 
    1b(ii) Multiple Modality
        -- get expected observations X
        --update_posterior_policies (just utility + state info gain) X
        --update_posterior_policies (now with param info gain) X

2 MULTIPLE FACTOR
- 2a Multiple factor, single timestep
    -- get expected states X
    2a(i) Single Modality
        -- get expected observations X
        --update_posterior_policies (just utility + state info gain) X
        --update_posterior_policies (now with param info gain) X
    2a(ii) Multiple Modality
        -- get expected observations X
        --update_posterior_policies (just utility + state info gain) X
        --update_posterior_policies (now with param info gain) X
- 2b Multiple factor, multiple timestep
    -- get expected states X
    2b(i) Single Modality
        -- get expected observations X
        --update_posterior_policies (just utility + state info gain) X
    2b(ii) Multiple Modality
        -- get expected observations X
        --update_posterior_policies (just utility + state info gain) X
"""

# %%
"""
1. SINGLE FACTOR
"""

"""
1(a) Single factor, single timestep test
"""

n_states = [3]
n_control = [3]

qs = Categorical(values = np.eye(*n_states)[0])

B = np.eye(*n_states)[:, :, np.newaxis]
B = np.tile(B, (1, 1, n_control[0]))
B = B.transpose(1, 2, 0)

B = Categorical(values = B)

n_step = 1
policies = core.construct_policies(n_states, n_control, policy_len=n_step, control_fac_idx=[0])

"""
1(a)(i) Single modality
"""

num_obs = [3]
num_modalities = len(num_obs)

A = Categorical(values = np.random.rand(*(num_obs + n_states)))
A.normalize()

C = Categorical(values = np.eye(*num_obs)[0])

q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=False,pA=None,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pA
pA = Dirichlet(values = np.ones((num_obs + n_states)))
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pB
pB = Dirichlet(values = np.ones(B.shape))
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

# %%
"""
1(a)(ii) Multiple modality
"""

num_obs = [3, 4]
num_modalities = len(num_obs)

A_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    A_numpy[modality] = np.random.rand((*([no] + n_states)))

A = Categorical(values = A_numpy)
A.normalize()

C_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    C_numpy[modality] = np.random.rand(no)

C = Categorical(values = C_numpy)
C.normalize()

q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=False,pA=None,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pA
pA_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    pA_numpy[modality] = np.ones((no, *n_states))

pA = Dirichlet(values = pA_numpy)
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pB
pB = Dirichlet(values = np.ones(B.shape))
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

# %%
"""
1(b) Single factor, multiple timestep test
"""
n_step = 3
policies = core.construct_policies(n_states, n_control, policy_len=n_step, control_fac_idx=[0])

"""
1(b)(i) Single modality
"""

num_obs = [3]
num_modalities = len(num_obs)

A = Categorical(values = np.random.rand(*(num_obs + n_states)))
A.normalize()
# qo_pi = core.get_expected_obs(qs_pi, A)

C = Categorical(values = np.eye(*num_obs)[0])

q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=False,pA=None,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pA
pA = Dirichlet(values = np.ones((num_obs + n_states)))
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pB
pB = Dirichlet(values = np.ones(B.shape))
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

# %%
"""
1(b)(ii) Multiple modality
"""

num_obs = [3, 4]
num_modalities = len(num_obs)

A_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    A_numpy[modality] = np.random.rand((*([no] + n_states)))

A = Categorical(values = A_numpy)
A.normalize()

# qo_pi = core.get_expected_obs(qs_pi, A)

C_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    C_numpy[modality] = np.random.rand(no)

C = Categorical(values = C_numpy)
C.normalize()

q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=False,pA=None,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pA
pA_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    pA_numpy[modality] = np.ones((no, *n_states))

pA = Dirichlet(values = pA_numpy)
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pB
pB = Dirichlet(values = np.ones(B.shape))
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

# %%
"""
2. MULTIPLE FACTOR
"""

"""
2(a) Multiple factor, single timestep test
"""

n_states = [3, 2]
n_control = [3, 2]
num_factors = len(n_states)

qs_numpy = np.empty(num_factors, dtype = object)
for factor, ns in enumerate(n_states):
    qs_numpy[factor] = np.random.rand(ns)

qs = Categorical(values = qs_numpy)
qs.normalize()

B_numpy = np.empty(num_factors, dtype = object)
for factor,nc in enumerate(n_control):
    tmp = np.eye(nc)[:, :, np.newaxis]
    tmp = np.tile(tmp, (1, 1, nc))
    B_numpy[factor] = tmp.transpose(1, 2, 0)

B = Categorical(values = B_numpy)

n_step = 1
policies = core.construct_policies(n_states, n_control, policy_len=n_step, control_fac_idx=[0,1])

"""
2(a)(i) Single modality
"""

num_obs = [3]
num_modalities = len(num_obs)

A = Categorical(values = np.random.rand(*(num_obs + n_states)))
A.normalize()

C = Categorical(values = np.eye(*num_obs)[0])

q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=False,pA=None,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pA
pA = Dirichlet(values = np.ones((num_obs + n_states)))
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pB
pB_numpy = np.empty(num_factors, dtype = object)
for factor,nc in enumerate(n_control):
    tmp = np.ones((nc,nc))[:, :, np.newaxis]
    tmp = np.tile(tmp, (1, 1, nc))
    pB_numpy[factor] = tmp.transpose(1, 2, 0)

pB = Dirichlet(values = pB_numpy)
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

# %%
"""
2(a)(i) Multiple modality
"""

num_obs = [3, 4]
num_modalities = len(num_obs)

A_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    A_numpy[modality] = np.random.rand((*([no] + n_states)))

A = Categorical(values = A_numpy)
A.normalize()

C_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    C_numpy[modality] = np.random.rand(no)

C = Categorical(values = C_numpy)
C.normalize()

q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=False,pA=None,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pA
pA_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    pA_numpy[modality] = np.ones((no, *n_states))

pA = Dirichlet(values = pA_numpy)
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pB
pB_numpy = np.empty(num_factors, dtype = object)
for factor,nc in enumerate(n_control):
    tmp = np.ones((nc,nc))[:, :, np.newaxis]
    tmp = np.tile(tmp, (1, 1, nc))
    pB_numpy[factor] = tmp.transpose(1, 2, 0)

# now do info gain about pB
pB = Dirichlet(values = pB_numpy)
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

# %%
"""
2(b) Multiple factor, multiple timestep test
"""
n_step = 3
policies = core.construct_policies(n_states, n_control, policy_len=n_step, control_fac_idx=[0])

# policy_i = policies[2]
# qs_pi = core.get_expected_states(qs, B, policy_i)
# qs_pi = core.get_expected_states(qs, B, policy_i, return_numpy=True)

"""
2(b)(i) Single modality
"""

num_obs = [3]
num_modalities = len(num_obs)

A = Categorical(values = np.random.rand(*(num_obs + n_states)))
A.normalize()

C = Categorical(values = np.eye(*num_obs)[0])

q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=False,pA=None,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pA
pA = Dirichlet(values = np.ones((num_obs + n_states)))
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pB
pB_numpy = np.empty(num_factors, dtype = object)
for factor,nc in enumerate(n_control):
    tmp = np.ones((nc,nc))[:, :, np.newaxis]
    tmp = np.tile(tmp, (1, 1, nc))
    pB_numpy[factor] = tmp.transpose(1, 2, 0)

pB = Dirichlet(values = pB_numpy)
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)

# %%
"""
2(b)(i) Multiple modality
"""

num_obs = [3, 4]
num_modalities = len(num_obs)

A_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    A_numpy[modality] = np.random.rand((*([no] + n_states)))

A = Categorical(values = A_numpy)
A.normalize()

C_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    C_numpy[modality] = np.random.rand(no)

C = Categorical(values = C_numpy)
C.normalize()

q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=False,pA=None,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pA
pA_numpy = np.empty(num_modalities, dtype = object)
for modality, no in enumerate(num_obs):
    pA_numpy[modality] = np.ones((no, *n_states))

pA = Dirichlet(values = pA_numpy)
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=None,gamma=16.0,return_numpy=True)

# now do info gain about pB
pB_numpy = np.empty(num_factors, dtype = object)
for factor,nc in enumerate(n_control):
    tmp = np.ones((nc,nc))[:, :, np.newaxis]
    tmp = np.tile(tmp, (1, 1, nc))
    pB_numpy[factor] = tmp.transpose(1, 2, 0)

# now do info gain about pB
pB = Dirichlet(values = pB_numpy)
q_pi = core.update_posterior_policies(qs, A, B, C, policies, use_utility=True, use_states_info_gain=True,
use_param_info_gain=True,pA=pA,pB=pB,gamma=16.0,return_numpy=True)


# %%
