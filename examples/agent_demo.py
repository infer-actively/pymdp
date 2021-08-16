import numpy as np
from pymdp.agent import Agent
from pymdp import utils
from pymdp.maths import softmax
import copy

obs_names = ["state_observation", "reward", "decision_proprioceptive"]
state_names = ["reward_level", "decision_state"]
action_names = ["uncontrolled", "decision_state"]

num_obs = [3, 3, 3]
num_states = [2, 3]
num_modalities = len(num_obs)
num_factors = len(num_states)

A = utils.obj_array_zeros([[o] + num_states for _, o in enumerate(num_obs)])

A[0][:, :, 0] = np.ones( (num_obs[0], num_states[0]) ) / num_obs[0]
A[0][:, :, 1] = np.ones( (num_obs[0], num_states[0]) ) / num_obs[0]
A[0][:, :, 2] = np.array([[0.8, 0.2], [0.0, 0.0], [0.2, 0.8]])

A[1][2, :, 0] = np.ones(num_states[0])
A[1][0:2, :, 1] = softmax(np.eye(num_obs[1] - 1)) # bandit statistics (mapping between reward-state (first hidden state factor) and rewards (Good vs Bad))
A[1][2, :, 2] = np.ones(num_states[0])

# establish a proprioceptive mapping that determines how the agent perceives its own `decision_state`
A[2][0,:,0] = 1.0
A[2][1,:,1] = 1.0
A[2][2,:,2] = 1.0

control_fac_idx = [1]
B = utils.obj_array(num_factors)
for f, ns in enumerate(num_states):
    B[f] = np.eye(ns)
    if f in control_fac_idx:
        B[f] = B[f].reshape(ns, ns, 1)
        B[f] = np.tile(B[f], (1, 1, ns))
        B[f] = B[f].transpose(1, 2, 0)
    else:
        B[f] = B[f].reshape(ns, ns, 1)

C = utils.obj_array_zeros(num_obs)
C[1][0] = 1.0  # put a 'reward' over first observation
C[1][1] = -2.0  # put a 'punishment' over first observation
# this implies that C[1][2] is 'neutral'

agent = Agent(A=A, B=B, C=C, control_fac_idx=[1])

# initial state
T = 5
o = [2, 2, 0]
s = [0, 0]

# transition/observation matrices characterising the generative process
A_gp = copy.deepcopy(A)
B_gp = copy.deepcopy(B)

for t in range(T):

    for g in range(num_modalities):
        print(f"{t}: Observation {obs_names[g]}: {o[g]}")

    qx = agent.infer_states(o)

    for f in range(num_factors):
        print(f"{t}: Beliefs about {state_names[f]}: {qx[f]}")

    agent.infer_policies()
    action = agent.sample_action()

    for f, s_i in enumerate(s):
        s[f] = utils.sample(B_gp[f][:, s_i, int(action[f])])

    for g, _ in enumerate(o):
        o[g] = utils.sample(A_gp[g][:, s[0], s[1]])
    
    print(np.argmax(s))
    print(f"{t}: Action: {action} / State: {s}")
