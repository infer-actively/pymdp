import os
import math
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.ndimage as ndimage
from pymdp.utils import fig2img
from equinox import field
from pymdp.envs.env import PymdpEnv


# load assets
assets_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
mouse_img = plt.imread(os.path.join(assets_dir, "mouse.png"))
right_mouse_img = jnp.clip(ndimage.rotate(mouse_img, 90, reshape=True), 0.0, 1.0)
left_mouse_img = jnp.clip(ndimage.rotate(mouse_img, -90, reshape=True), 0.0, 1.0)
up_mouse_img = jnp.clip(ndimage.rotate(mouse_img, 180, reshape=True), 0.0, 1.0)
cheese_img = plt.imread(os.path.join(assets_dir, "cheese.png"))
shock_img = plt.imread(os.path.join(assets_dir, "shock.png"))


class TMaze(PymdpEnv):
    """
    Implementation of the T-Maze environment where an agent must navigate to find a reward, with:
    - 4 locations: centre, left arm, right arm, and cue position (bottom)
    - 2 reward conditions: reward in left or right arm
    - Cues embedded in location observations when at cue position
    """
    reward_condition: float = field(static=True)
    cue_validity: float = field(static=True)
    reward_probability: float = field(static=True)
    dependent_outcomes: bool = field(static=True)
    punishment_probability: float = field(static=True)

    def __init__(self, reward_condition=None, cue_validity=0.95, reward_probability=1.0, dependent_outcomes=False, punishment_probability=1.0):
        """
        Initialise T-Maze environment. A is the observation likelihood matrix, B is the transition matrix, D is the initial state distribution.
        Args:
            reward_condition: If specified, fixes reward to left (0) or right (1) arm, otherwise reward is randomly assigned
            cue_validity: Probability of cue correctly indicating reward location
            reward_probability: Probability of getting reward in correct arm
            dependent_outcomes: If True, punishment occurs as a function of reward probability (i.e., if reward probability is 0.8, then 20% punishment). If False, punishment occurs with set probability (i.e., 20% no outcome and punishment will only occur in the other (non-rewarding) arm defined by the punishment_probability parameter)
            punishment_probability: Probability of getting punishment in incorrect arm
        """
        self.reward_condition = reward_condition
        self.cue_validity = cue_validity
        self.reward_probability = reward_probability
        self.dependent_outcomes = dependent_outcomes
        self.punishment_probability = punishment_probability

        # generate observation likelihood(A), state transition (B), and initial state (D) tensors
        A, A_dependencies = self.generate_A()
        B, B_dependencies = self.generate_B()
        D = self.generate_D()

        super().__init__(A=A, B=B, D=D, A_dependencies=A_dependencies, B_dependencies=B_dependencies)

    def generate_A(self):
        """
        Generate observation likelihood tensors.
        
        Returns:
        A[0]: Location observations (5x4x2)
            - Maps true location to observed location with cue information
            - [centre, left, right, cued left, cued right] x [centre loc, left loc, right loc, cue loc] x [reward in left arm, reward in right arm]]
        A[1]: Outcome observations (3x4x2)
            - [no outcome, reward, punishment] x [centre loc, left loc, right loc, cue loc] x [reward in left arm, reward in right arm]]
        """
        A = []
        A.append(jnp.zeros([5, 4, 2]))  # observation modality 0 of the locations and cues
        A.append(jnp.zeros([3, 4, 2]))  # observation modality 1 of the outcomes

        A_dependencies = [[0, 1], [0, 1]]

        
        for loc in range(4): # for each location: [centre, left, right, cue]
            for reward_condition in range(2): # for each reward condition: [left, right]
                if loc == 0: # when at starting location (centre)
                    # observe centre
                    A[0] = A[0].at[0, loc, reward_condition].set(1.0)
                    # observe no outcome
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)

                elif loc == 3: # when at cue location
                    # observe cued left (index 3) or cued right (index 4) based on reward condition and cue validity
                    if reward_condition == 0:  # reward in left arm
                        # correctly cued left arm with cue_validity probability
                        A[0] = A[0].at[3, loc, reward_condition].set(self.cue_validity)
                        # incorrectly cued right arm with (1 - cue_validity) probability
                        A[0] = A[0].at[4, loc, reward_condition].set(1 - self.cue_validity)
                    else:  # reward in right arm
                        # correctly cued right arm with cue_validity probability
                        A[0] = A[0].at[4, loc, reward_condition].set(self.cue_validity)
                        # incorrectly cued left arm with (1 - cue_validity) probability
                        A[0] = A[0].at[3, loc, reward_condition].set(1 - self.cue_validity)
                    
                    # observe no outcome at cue location
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)

                else: # when at left (loc == 1) or right (loc == 2) arm
                    # observe the actual location (left arm = 1, right arm = 2)
                    A[0] = A[0].at[loc, loc, reward_condition].set(1.0)
                    
                    if loc == (reward_condition + 1):  # if at correct reward location
                        # observe reward with probability reward_probability, otherwise no outcome or punishment depending on dependent_outcomes
                        A[1] = A[1].at[1, loc, reward_condition].set(self.reward_probability)
                        if self.dependent_outcomes:
                            # if dependent, remaining probability goes to punishment
                            A[1] = A[1].at[2, loc, reward_condition].set(1 - self.reward_probability)
                        else:
                            # if independent, remaining probability goes to no outcome
                            A[1] = A[1].at[0, loc, reward_condition].set(1 - self.reward_probability)
                    else: # if at incorrect reward location
                        if self.dependent_outcomes:
                            # if dependent, probabilities are flipped from above
                            A[1] = A[1].at[2, loc, reward_condition].set(self.reward_probability)
                            A[1] = A[1].at[1, loc, reward_condition].set(1 - self.reward_probability)
                        else:
                            # if independent, punishment occurs with set probability
                            A[1] = A[1].at[2, loc, reward_condition].set(self.punishment_probability)
                            A[1] = A[1].at[0, loc, reward_condition].set(1 - self.punishment_probability)

        return A, A_dependencies

    def generate_B(self):
        """
        Generate state transition tensors.
        
        Returns:
        B[0]: Location transitions (4x4x4)
            - Agent can move from any location to any other location (no adjacency constraints)
            - [centre, left, right, cue] x [centre, left, right, cue] x [move to centre, move to left, move to right, move to cue]
        B[1]: Reward condition transitions (2x2x1)
            - Reward location stays fixed
            - [left, right] x [left, right] x [null dimension for pymdp data structure requirements]
        """
        B = []
        
        num_locations = 4  # 0: centre, 1: left, 2: right, 3: cue
        B_loc = jnp.zeros((num_locations, num_locations, num_locations))
        
        # allow movement from any location to any location
        for _from in range(num_locations):
            for _to in range(num_locations):
                for action in range(num_locations):
                    if action == _to:  # trying to move to location '_to'
                        # allow movement to the target location
                        B_loc = B_loc.at[_to, _from, action].set(1.0)
                    # if not trying to move to location '_to', stay in current location - this is handled implicitly since we only set transitions when action == _to
        B.append(B_loc)

        # reward condition stays fixed
        B_reward = jnp.eye(2).reshape(2, 2, 1)
        B.append(B_reward)

        B_dependencies = [[0], [1]]

        return B, B_dependencies

    def generate_D(self):
        """
        Define experimental conditions and initial environmental states.
        
        Returns two state vectors:
        D[0]: Initial location (4,)
            - Agent always starts in centre (index 0)
        D[1]: Initial reward condition (2,)
            - Either random (50/50) or fixed based on reward_condition
        """
        D = []
        D_loc = jnp.zeros([4])
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

    def render(self, observations, mode="human"):
        batch_size = observations[0].shape[0]
     
        plt.clf()  # clear the current figure

        # create n x n subplots for the batch_size
        n = math.ceil(math.sqrt(batch_size))

        # create the subplots
        fig, axes = plt.subplots(n, n, figsize=(5,5))

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
            ax.pcolormesh(
                X, Y, jnp.ones(grid_dims), edgecolors="none", vmin=0, vmax=30, linewidth=5, cmap="coolwarm", snap=True
            )
            ax.invert_yaxis()
            ax.axis("off")
            ax.set_aspect("equal")

            ax.add_patch(
                patches.Rectangle(
                    (0, 1),
                    1.0,
                    2.0,
                    linewidth=0,
                    facecolor=[1.0, 1.0, 1.0],
                )
            )

            ax.add_patch(
                patches.Rectangle(
                    (2, 1),
                    1.0,
                    2.0,
                    linewidth=0,
                    facecolor=[1.0, 1.0, 1.0],
                )
            )

            ax.add_patch(
                patches.Rectangle(
                    (0, 0),
                    1.0,
                    1.0,
                    linewidth=0,
                    facecolor="tab:orange",
                )
            )

            ax.add_patch(
                patches.Rectangle(
                    (2, 0),
                    1.0,
                    1.0,
                    linewidth=0,
                    facecolor="tab:purple",
                )
            )

            # show the cue based on location observation
            loc_obs = observations[0][i, 0]
            if loc_obs == 3:  # cued left arm
                cue_color = "tab:orange"
                edge_color = "tab:gray"
            elif loc_obs == 4:  # cued right arm
                cue_color = "tab:purple"
                edge_color = "tab:gray"
            else:  # no cue
                cue_color = "tab:gray"
                edge_color = "tab:gray"

            ax.add_patch(
                patches.Circle(
                    (1.5, 2.5),
                    0.3,
                    linewidth=8,
                    edgecolor=edge_color,
                    facecolor=cue_color,
                )
            )

            # show the reward
            loc_obs = observations[0][i, 0]
            reward = observations[1][i, 0]
            
            # determine reward coordinates based on location observation
            if loc_obs == 1:  # left arm
                coords = (0.5, 0.5)
            elif loc_obs == 2:  # right arm
                coords = (2.5, 0.5)
            else:
                coords = None  # no reward shown in other locations

            # only show reward/punishment if in arm locations and there's an outcome
            if coords is not None:
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

            # show the mouse based on location observation
            if loc_obs == 0:
                # centre
                up_mouse_im = OffsetImage(up_mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(up_mouse_im, (1.5, 1.5), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)
            elif loc_obs == 1:
                # left arm
                left_mouse_im = OffsetImage(left_mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(left_mouse_im, (0.75, 0.5), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)
            elif loc_obs == 2:
                # right arm
                right_mouse_im = OffsetImage(right_mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(right_mouse_im, (2.25, 0.5), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)
            elif loc_obs == 3 or loc_obs == 4:
                # cue location (bottom) - either cued left or cued right
                down_mouse_im = OffsetImage(mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(down_mouse_im, (1.5, 2.25), xycoords="data", frameon=False)
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
