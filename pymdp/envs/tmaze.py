import os
import math
import jax.numpy as jnp

import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.ndimage as ndimage
from pymdp.utils import fig2img

from equinox import field

from .env import Env

# load assets
assets_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
mouse_img = plt.imread(os.path.join(assets_dir, "mouse.png"))
right_mouse_img = jnp.clip(ndimage.rotate(mouse_img, 90, reshape=True), 0.0, 1.0)
left_mouse_img = jnp.clip(ndimage.rotate(mouse_img, -90, reshape=True), 0.0, 1.0)
up_mouse_img = jnp.clip(ndimage.rotate(mouse_img, 180, reshape=True), 0.0, 1.0)
cheese_img = plt.imread(os.path.join(assets_dir, "cheese.png"))
shock_img = plt.imread(os.path.join(assets_dir, "shock.png"))


class TMaze(Env):
    """
    Implementation of the 3-arm T-Maze environment.
    A T-shaped maze where an agent must navigate to find a reward, with:
    - 4 locations: center, left arm, right arm, and cue position (bottom arm) 
    - 2 reward conditions: reward in left or right arm
    - Cues that indicate which arm contains the reward
    """

    reward_probability: float = field(static=True)

    def __init__(self, batch_size=1, reward_probability=0.98, reward_condition=None):

        """
        Initialize T-Maze environment. A is the observation likelihood matrix, B is the transition matrix, D is the initial state distribution.
        Args:
            batch_size: Number of parallel environments
            reward_probability: Probability of getting reward/punishment in correct/incorrect arm
            reward_condition: If specified, fixes reward to left (0) or right (1) arm, otherwise reward is randomly assigned
        """
        self.reward_probability = reward_probability

        # Generate and broadcast observation likelihood(A), transition (B), and initial state (D) tensors to the batch size
        A, A_dependencies = self.generate_A()
        A = [jnp.broadcast_to(a, (batch_size,) + a.shape) for a in A]
        B, B_dependencies = self.generate_B()
        B = [jnp.broadcast_to(b, (batch_size,) + b.shape) for b in B]
        D = self.generate_D(reward_condition)
        D = [jnp.broadcast_to(d, (batch_size,) + d.shape) for d in D]

        params = {
            "A": A,
            "B": B,
            "D": D,
        }

        dependencies = { # specifying which matrix is dependent on which state factors allows you to not have to specify all combinations of state factors in the matrix
            "A": A_dependencies, 
            "B": B_dependencies,
        }

        super().__init__(params, dependencies)

    def generate_A(self):
        """
        Generate observation likelihood tensors.
        
        Returns three observation matrices:
        A[0]: Location observations (4x4 identity matrix)
            - Maps true location to observed location
        A[1]: Reward observations (3x4x2 matrix)
            - [no outcome, reward, punishment] x [4 locations] x [2 reward conditions]
        A[2]: Cue observations (3x4x2 matrix)
            - [no cue, cued left arm, cued right arm] x [4 locations] x [2 reward conditions]
        """
        A = []
        A.append(jnp.eye(4))
        A.append(jnp.zeros([3, 4, 2]))
        A.append(jnp.zeros([3, 4, 2]))

        A_dependencies = [[0], [0, 1], [0, 1]]

        
        for loc in range(4): # for each location: [center, left, right, cue]
            for reward_condition in range(2): # for each reward condition: [left, right]
                if loc == 0: # when at starting location, the centre location, there is no reward and no cue
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)
                    A[2] = A[2].at[0, loc, reward_condition].set(1.0)

                elif loc == 3: # when at cue location, the cue indicates the reward condition unambiguously but there is no reward
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)
                    A[2] = A[2].at[reward_condition + 1, loc, reward_condition].set(1.0)

                else: # when at one of the outcome (reward or punishment) arms
                    if loc == (reward_condition + 1): # if the agent is at the correct reward location
                        high_prob_idx = 1 # there is a high probability of reward
                        low_prob_idx = 2 # there is a low probability of punishment
                    else:
                        high_prob_idx = 2 # otherwise, there is a high probability of punishment   
                        low_prob_idx = 1 # there is a low probability of reward

                    A[1] = A[1].at[high_prob_idx, loc, reward_condition].set(self.reward_probability)
                    A[1] = A[1].at[low_prob_idx, loc, reward_condition].set(1 - self.reward_probability)

                    A[2] = A[2].at[0, loc, reward_condition].set(1.0) # there are no cues in the outcome arms

        return A, A_dependencies

    def generate_B(self):
        """
        Generate transition model matrices.
        
        Returns two transition matrices:
        B[0]: Location transitions (4x4x4)
            - Agent can teleport between any locations
        B[1]: Reward condition transitions (2x2x1)
            - Reward location stays fixed
        """
        B = []

        # agent can move from any location to any location according to the environment
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
        Generate initial state distribution.
        
        Returns two initial state vectors:
        D[0]: Initial location (4,)
            - Agent always starts in center (index 0)
        D[1]: Initial reward condition (2,)
            - Either random (50/50) or fixed based on reward_condition
        
        Args:
            reward_condition: If specified, fixes reward to left (0) or right (1) arm, otherwise random
        """
        D = []
        D_loc = jnp.zeros([4])
        D_loc = D_loc.at[0].set(1.0) # the agent always starts at the centre
        D.append(D_loc)

        if reward_condition is None:
            # 50/50 chance of reward in left/right arm
            D_reward = jnp.ones(2) * 0.5
        else:
            # reward is fixed in the left/right arm
            D_reward = jnp.zeros(2)
            D_reward = D_reward.at[reward_condition].set(1.0)
        D.append(D_reward)
        return D

    def render(self, mode="human"):
        batch_size = self.params["A"][0].shape[0]

        # Create n x n subplots for the batch_size
        n = math.ceil(math.sqrt(batch_size))

        # Create the subplots
        fig, axes = plt.subplots(n, n, figsize=(6, 6))

        # Loop through the batch_size and plot on each subplot
        for i in range(batch_size):
            row = i // n
            col = i % n
            if batch_size == 1:
                ax = axes
            else:
                ax = axes[row, col]

            grid_dims = [3, 3]
            X, Y = jnp.meshgrid(jnp.arange(grid_dims[1] + 1), jnp.arange(grid_dims[0] + 1))
            h = ax.pcolormesh(
                X, Y, jnp.ones(grid_dims), edgecolors="none", vmin=0, vmax=30, linewidth=5, cmap="coolwarm", snap=True
            )
            ax.invert_yaxis()
            ax.axis("off")
            ax.set_aspect("equal")

            edge_left = ax.add_patch(
                patches.Rectangle(
                    (0, 1),
                    1.0,
                    2.0,
                    linewidth=0,
                    facecolor=[1.0, 1.0, 1.0],
                )
            )

            edge_right = ax.add_patch(
                patches.Rectangle(
                    (2, 1),
                    1.0,
                    2.0,
                    linewidth=0,
                    facecolor=[1.0, 1.0, 1.0],
                )
            )

            arm_left = ax.add_patch(
                patches.Rectangle(
                    (0, 0),
                    1.0,
                    1.0,
                    linewidth=0,
                    facecolor="tab:orange",
                )
            )

            arm_right = ax.add_patch(
                patches.Rectangle(
                    (2, 0),
                    1.0,
                    1.0,
                    linewidth=0,
                    facecolor="tab:purple",
                )
            )

            # show the cue
            cue = self.current_obs[2][i, 0]
            if cue == 0:
                cue_color = "tab:gray"
            elif cue == 1:
                # left
                cue_color = "tab:orange"
            elif cue == 2:
                # right
                cue_color = "tab:purple"

            cue = ax.add_patch(
                patches.Circle(
                    (1.5, 2.5),
                    0.3,
                    linewidth=0,
                    facecolor=cue_color,
                )
            )

            # show the reward
            loc = self.current_obs[0][i, 0]

            reward = self.current_obs[1][i, 0]

            if loc == 1:
                coords = (0.5, 0.5)
            elif loc == 2:
                coords = (2.5, 0.5)

            if reward == 1:
                # cheese
                cheese_im = OffsetImage(cheese_img, zoom=0.025 / n)
                ab_cheese = AnnotationBbox(cheese_im, coords, xycoords="data", frameon=False)
                an_cheese = ax.add_artist(ab_cheese)
                an_cheese.set_zorder(2)

            elif reward == 2:
                # shock
                shock_im = OffsetImage(shock_img, zoom=0.1 / n)
                ab_shock = AnnotationBbox(shock_im, coords, xycoords="data", frameon=False)
                ab_shock = ax.add_artist(ab_shock)
                ab_shock.set_zorder(2)

            # show the mouse
            if loc == 0:
                # center
                up_mouse_im = OffsetImage(up_mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(up_mouse_im, (1.5, 1.5), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)
            elif loc == 1:
                # left
                left_mouse_im = OffsetImage(left_mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(left_mouse_im, (0.75, 0.5), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)
            elif loc == 2:
                # right
                right_mouse_im = OffsetImage(right_mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(right_mouse_im, (2.25, 0.5), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)
            elif loc == 3:
                # bottom
                down_mouse_im = OffsetImage(mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(down_mouse_im, (1.5, 2.25), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)

        # Hide any extra subplots if batch_size isn't a perfect square
        for i in range(batch_size, n * n):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            return fig2img(fig)
