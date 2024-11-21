import os
import math
import jax.numpy as jnp
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
    - 4 locations: centre, left arm, right arm, and cue position (bottom arm) 
    - 2 reward conditions: reward in left or right arm
    - Cues that indicate which arm contains the reward
    """

    reward_probability: float = field(static=True)
    punishment_probability: float = field(static=True)
    cue_validity: float = field(static=True)
    reward_condition: float = field(static=True)

    def __init__(self, batch_size=1, reward_probability=1.0, punishment_probability=1.0, cue_validity=0.95, reward_condition=None):
        """
        Initialize T-Maze environment. A is the observation likelihood matrix, B is the transition matrix, D is the initial state distribution.
        Args:
            batch_size: Number of parallel environments
            reward_probability: Probability of getting reward in correct arm
            punishment_probability: Probability of getting punishment in incorrect arm
            cue_validity: Probability of cue correctly indicating reward location
            reward_condition: If specified, fixes reward to left (0) or right (1) arm, otherwise reward is randomly assigned
        """
        self.reward_probability = reward_probability
        self.punishment_probability = punishment_probability
        self.cue_validity = cue_validity
        self.reward_condition = reward_condition

        # Generate and broadcast observation likelihood(A), transition (B), and initial state (D) tensors to the batch size
        A, A_dependencies = self.generate_A()
        A = [jnp.broadcast_to(a, (batch_size,) + a.shape) for a in A]
        B, B_dependencies = self.generate_B()
        B = [jnp.broadcast_to(b, (batch_size,) + b.shape) for b in B]
        D = self.generate_D()
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
        A[0]: Location observations (5x5 identity matrix)
            - Maps true location to observed location
            - [centre, left, right, cue, middle] x [centre, left, right, cue, middle]
        A[1]: Reward observations (3x5 x2 matrix)
            - [no outcome, reward, punishment] x [5 locations] x [2 reward conditions]
        A[2]: Cue observations (3x5x2 matrix)
            - [no cue, cued left arm, cued right arm] x [5 locations] x [2 reward conditions]
        """
        A = []
        A.append(jnp.eye(5))
        A.append(jnp.zeros([3, 5, 2]))
        A.append(jnp.zeros([3, 5, 2]))

        A_dependencies = [[0], [0, 1], [0, 1]]

        
        for loc in range(5): # for each location: [centre, left, right, cue, middle]
            for reward_condition in range(2): # for each reward condition: [left, right]
                if loc == 0: # when at starting location (centre), there is no reward and no cue
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)
                    A[2] = A[2].at[0, loc, reward_condition].set(1.0)

                elif loc == 3: # when at cue location
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)  # still no reward at cue location
                    
                    # set probability for correct cue based on cue_validity
                    A[2] = A[2].at[reward_condition + 1, loc, reward_condition].set(self.cue_validity)
                    # set probability for incorrect cue
                    wrong_cue = (1 - reward_condition) + 1  # if reward_condition is 0 (left), wrong_cue is 2 (right); if reward_condition is 1 (right), wrong_cue is 1 (left)
                    A[2] = A[2].at[wrong_cue, loc, reward_condition].set(1 - self.cue_validity)

                elif loc == 4: # when at middle location
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0) # there are no outcomes at the middle
                    A[2] = A[2].at[0, loc, reward_condition].set(1.0) # there are no cues at the middle

                else: # when at one of the outcome (reward or punishment) arms
                    if loc == (reward_condition + 1): # if the agent is at the correct reward location
                        # in correct arm: probability of reward, otherwise no outcome
                        A[1] = A[1].at[1, loc, reward_condition].set(self.reward_probability)
                        A[1] = A[1].at[0, loc, reward_condition].set(1 - self.reward_probability)
                    else:
                       # in incorrect arm: probability of punishment, otherwise no outcome
                        A[1] = A[1].at[2, loc, reward_condition].set(self.punishment_probability)
                        A[1] = A[1].at[0, loc, reward_condition].set(1 - self.punishment_probability)

                    A[2] = A[2].at[0, loc, reward_condition].set(1.0) # there are no cues in the outcome arms

        return A, A_dependencies

    def generate_B(self):
        """
        Generate transition model matrices.
        
        Returns two transition matrices:
        B[0]: Location transitions (5x5x5)
            - Agent can move between adjacent locations in the T-maze
        B[1]: Reward condition transitions (2x2x1)
            - Reward location stays fixed
        """
        B = []

        
        num_locations = 5  # 0: centre, 1: left, 2: right, 3: cue, 4: middle
        B_loc = jnp.zeros((num_locations, num_locations, num_locations))
        
        # defining valid (adjacent) connections in the T-maze
        valid_connections = [ # (from_location, to_location)
            (0, 3),  # centre <-> cue
            (3, 0),
            (0, 4),  # centre <-> middle
            (4, 0),
            (4, 1),  # middle <-> left arm
            (1, 4),
            (4, 2),  # middle <-> right arm
            (2, 4),
        ]
        
        # filling in the transition matrix
        for _from in range(num_locations):
            for _to in range(num_locations):
                for action in range(num_locations):
                    if action == _to:  # trying to move to location '_to'
                        if (_from, _to) in valid_connections:  # if movement is valid
                            B_loc = B_loc.at[_to, _from, action].set(1.0)
                        else:  # if movement is invalid, stay in current location
                            B_loc = B_loc.at[_from, _from, action].set(1.0)
        
        B.append(B_loc)

        # reward condition stays fixed
        B_reward = jnp.eye(2).reshape(2, 2, 1)
        B.append(B_reward)

        B_dependencies = [[0], [1]]

        return B, B_dependencies

    def generate_D(self):
        """
        Generate initial state distribution.
        
        Returns two initial state vectors:
        D[0]: Initial location (5,)
            - Agent always starts in centre (index 0)
        D[1]: Initial reward condition (2,)
            - Either random (50/50) or fixed based on reward_condition
        
        Args:
            reward_condition: If specified, fixes reward to left (0) or right (1) arm, otherwise random
        """
        D = []
        D_loc = jnp.zeros([5])
        D_loc = D_loc.at[0].set(1.0) # the agent always starts at the centre
        D.append(D_loc)

        if self.reward_condition is None:
            # 50/50 chance of reward in left/right arm
            D_reward = jnp.ones(2) * 0.5
        else:
            # reward is fixed in the left/right arm
            D_reward = jnp.zeros(2)
            D_reward = D_reward.at[self.reward_condition].set(1.0)
        D.append(D_reward)
        return D

    def render(self, mode="human", observations=None):
        if observations is not None:
            current_obs = observations
            batch_size = observations[0].shape[0]
        else:
            current_obs = self.current_obs
            batch_size = self.params["A"][0].shape[0]

        plt.clf()  # Clear the current figure
        

        # create n x n subplots for the batch_size
        n = math.ceil(math.sqrt(batch_size))

        # create the subplots
        fig, axes = plt.subplots(n, n, figsize=(6, 6))

        # loop through the batch_size and plot on each subplot
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
            cue = current_obs[2][i, 0]
            if cue == 0:
                cue_color = "tab:gray"
                edge_color = "tab:gray"
            elif cue == 1:
                # left
                cue_color = "tab:orange"
                edge_color = "tab:gray"
            elif cue == 2:
                # right
                cue_color = "tab:purple"
                edge_color = "tab:gray"

            cue = ax.add_patch(
                patches.Circle(
                    (1.5, 2.5),
                    0.3,
                    linewidth=8,
                    edgecolor=edge_color,
                    facecolor=cue_color,
                )
            )

            # show the reward
            loc = current_obs[0][i, 0]
            reward = current_obs[1][i, 0]
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
                # centre
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
            elif loc == 4:
                # middle
                middle_mouse_im = OffsetImage(up_mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(middle_mouse_im, (1.5, 0.5), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)

        # hide any extra subplots if batch_size isn't a perfect square
        for i in range(batch_size, n * n):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            img = fig2img(fig)
            plt.close(fig) 
            return img
