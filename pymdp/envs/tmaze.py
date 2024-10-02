import jax.numpy as jnp
from equinox import field

from pympd.envs.env import Env


class TMaze(Env):
    """
    Implementation of the 3-arm T-Maze environment.
    """
    reward_probability: float = field(static=True)

    def __init__(self, batch_size=1, reward_probability=0.98, reward_condition=None):
        self.reward_probability = reward_probability

        A, A_dependencies = self.generate_A()
        A = [jnp.broadcast_to(a, (batch_size,) + a.shape) for a in A]
        B, B_dependencies = self.generate_B()
        B = [jnp.broadcast_to(b, (batch_size,) + b.shape) for b in B]
        D = self.generate_D(reward_condition)

        params = {
            "A": A,
            "B": B,
            "D": D,
        }

        dependencies = {
            "A": A_dependencies,
            "B": B_dependencies,
        }

        super().__init__(params, dependencies)


    def generate_A(self):
        """
            T-maze has 3 observation modalities: location, reward and cue,
            and 2 state factors: agent location [center, left, right, cue] and reward location [left, right]
        """
        A = []
        A.append(jnp.eye(4))
        A.append(jnp.zeros([2, 4, 2]))
        A.append(jnp.zeros([2, 4, 2]))

        A_dependencies = [[0], [0, 1], [0, 1]]

        # 4 locations : [center, left, right, cue]
        for loc in range(4):
            # 2 reward conditions: [left, right]
            for reward_condition in range(2):
                # start location
                if loc == 0:
                    # When in the centre location, reward observation is always 'no reward'
                    # or the outcome with index 0
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)

                    # When in the centre location, cue is totally ambiguous with respect to the reward condition
                    A[2] = A[2].at[:, loc, reward_condition].set(0.5)

                # The case when loc == 3, or the cue location ('bottom arm')
                elif loc == 3:

                    # When in the cue location, reward observation is always 'no reward'
                    # or the outcome with index 0
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)

                    # When in the cue location, the cue indicates the reward condition umambiguously
                    # signals where the reward is located
                    A[2] = A[2].at[reward_condition, loc, reward_condition].set(1.0)

                # The case when the agent is in one of the (potentially) rewarding arms
                else:

                    # When location is consistent with reward condition
                    if loc == (reward_condition + 1):
                        # Means highest probability is concentrated over reward outcome
                        high_prob_idx = 1
                        # Lower probability on loss outcome
                        low_prob_idx = 2
                    else:
                        # Means highest probability is concentrated over loss outcome
                        high_prob_idx = 2
                        # Lower probability on reward outcome
                        low_prob_idx = 1

                    A[1] = A[1].at[high_prob_idx, loc, reward_condition].set(self.reward_probility)
                    A[1] = A[1].at[low_prob_idx, loc, reward_condition].set(1 - self.reward_probability)

                    # Cue is ambiguous when in the reward location
                    A[2] = A[2].at[:, loc, reward_condition].set(0.5)

        return A, A_dependencies


    def generate_B(self):
        """
            T-maze has 2 state factors: 
            agent location [center, left, right, cue] and reward location [left, right]
            agent can move between locations by teleporting, reward location stays fixed
        """
        B = []

        # agent can teleport to any location
        B_loc = jnp.eye(4)    
        B_loc = B_loc.reshape(4, 4, 1)
        B_loc = jnp.tile(B_loc, (1, 1, 4))
        B_loc = B_loc.transpose(1, 2, 0)
        B.append(B_loc)

        # reward condition stays fixed
        B_reward = jnp.eye(2).reshape(2, 2, 1)
        B.append(B_reward)

        B_dependencies = [[0], [1]]

        return B, B_dependencies
    
    def generate_D(self, reward_condition=None):
        """
            Agent starts at center
            Reward condition can be set or randomly sampled
        """
        D = []
        D_loc = jnp.zeros([4])
        D_loc = D_loc.at[0].set(1.0)
        D.append(D_loc)

        if reward_condition is None:
            D_reward = jnp.ones(2) * 0.5
        else:
            D_reward = jnp.zeros(2)
            D_reward = D_reward.at[reward_condition].set(1.0)
        D.append(D_reward)
        return D