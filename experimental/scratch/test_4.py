import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from pymdp.agent import Agent
from pymdp.envs import TMazeEnv

reward_probabilities = [0.6, 0.4]
# reward_probabilities = [0.98, 0.02] # probabilities used in Karl's original SPM demo

env = TMazeEnv(reward_probs = reward_probabilities)
agent = Agent(A=env.get_likelihood_dist(), B=env.get_transition_dist(), control_fac_idx=[0])
agent.C[1][1] = 3.0
agent.C[1][2] = -3.0
T = 10

obs = env.reset()

reward_conditions = ["Reward on Left", "Reward on Right"]
msg = """ === Starting experiment === \n Reward condition: {}, Initial observation {} """
print(msg.format(reward_conditions[env.reward_condition], obs))

for t in range(T):
    qx = agent.infer_states(obs)
    msg = """[{}] Inference [Arm {} / reward {}] """
    print(msg.format(t, qx[0].sample(), qx[1].sample(), obs[0], obs[1]))

    q_pi, efe = agent.infer_policies()

    action = agent.sample_action()

    msg = """[Step {}] Action: [Move to Arm {}]"""
    print(msg.format(t, action[0]))

    obs = env.step(action)

    msg = """[Step {}] Observation: [Arm {}, Reward {}]"""
    print(msg.format(t, obs[0], obs[1]))

