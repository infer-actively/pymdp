
import numpy as np
from pymdp.agent import Agent
from pymdp.core import utils
from pymdp.core.maths import softmax
from pymdp.distributions import Categorical, Dirichlet
import copy

num_obs = [3,3]
num_modalities = len(num_obs)

num_states = [2, 2]
num_factors = len(num_states)

A = utils.obj_array(num_modalities)
for g, o in enumerate(num_obs):
    A[g] = np.zeros([o] + num_states)

A[0][:,0,0] = softmax(utils.onehot(0,num_obs[0]))
A[0][:,1,0] = softmax(np.array([0.0, 1.0, 1.0]))
A[0][:,:,1] = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])


A[1][0:2,:,0] = softmax(np.eye(num_obs[1]-1))
A[1][2,:,1] = np.ones(num_states[0])

B = utils.obj_array(num_factors)
for f, ns in enumerate(num_states):
    B[f] = np.eye(ns)
    if f == 0:
        B[f] = B[f].reshape(ns, ns, 1)
        B[f] = np.tile(B[f], (1, 1, ns))
        B[f] = B[f].transpose(1,2,0)
    elif f == 1:
        B[f] = B[f].reshape(ns,ns,1)


C = utils.obj_array(num_modalities)
C[0] = np.zeros(num_obs[0])
C[1] = np.zeros(num_obs[1])
C[1][0] = 1.0 # put a 'reward' over first observation 
C[1][1] = -5.0 # put a 'punishment' over first observation 
# this implies that C[1][2] is 'neutral'

agent = Agent(A = A, B = B, C = C, control_fac_idx = [0])

T = 5

o = [2, 2]
s = [1, 0]

# the transition/observation matrices characterising the generative process
A_gp = copy.deepcopy(A)
B_gp = copy.deepcopy(B)

for t in range(T):

    qx = agent.infer_states(tuple(o))
    print(f"Beliefs about first factor: {qx[0].values}")
    print(f"Beliefs about second factor: {qx[1].values}")

    agent.infer_policies()

    action = agent.sample_action()

    for f, s_i in enumerate(s):
        ps = Categorical(values = B_gp[0][:,s_i,action[f]]) # distribution over next state
        s[f] = ps.sample() # sample from state distribution

    for g, _ in enumerate(o):
        po = Categorical(values = A_gp[g][:,s[0],s[1]])
        o[g] = po.sample()

    print(f"Action: {action}")
    print(f"State: {s}")
    print(f"Observation: {o}")

