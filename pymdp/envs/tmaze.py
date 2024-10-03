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
    """

    reward_probability: float = field(static=True)

    def __init__(self, batch_size=1, reward_probability=0.98, reward_condition=None):
        self.reward_probability = reward_probability

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

        dependencies = {
            "A": A_dependencies,
            "B": B_dependencies,
        }

        super().__init__(params, dependencies)

    def generate_A(self):
        """
        T-maze has 3 observation modalities:
            location: [center, left, right, cue],
            reward [no reward, reward, punishment]
            and cue [no clue, left arm, right arm],
        and 2 state factors: agent location [center, left, right, cue] and reward location [left, right]
        """
        A = []
        A.append(jnp.eye(4))
        A.append(jnp.zeros([3, 4, 2]))
        A.append(jnp.zeros([3, 4, 2]))

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

                    # When in the centre location, cue is absent
                    A[2] = A[2].at[0, loc, reward_condition].set(1.0)

                # The case when loc == 3, or the cue location ('bottom arm')
                elif loc == 3:

                    # When in the cue location, reward observation is always 'no reward'
                    # or the outcome with index 0
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)

                    # When in the cue location, the cue indicates the reward condition umambiguously
                    # signals where the reward is located
                    A[2] = A[2].at[reward_condition + 1, loc, reward_condition].set(1.0)

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

                    A[1] = A[1].at[high_prob_idx, loc, reward_condition].set(self.reward_probability)
                    A[1] = A[1].at[low_prob_idx, loc, reward_condition].set(1 - self.reward_probability)

                    # Cue is absent here
                    A[2] = A[2].at[0, loc, reward_condition].set(1.0)

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
