import os
import math
from typing import Optional
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import scipy.ndimage as ndimage
from pymdp.utils import fig2img
from equinox import field
from .env import PymdpEnv


# load assets
assets_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
mouse_img = plt.imread(os.path.join(assets_dir, "mouse.png"))
right_mouse_img = jnp.clip(ndimage.rotate(mouse_img, 90, reshape=True), 0.0, 1.0)
left_mouse_img = jnp.clip(ndimage.rotate(mouse_img, -90, reshape=True), 0.0, 1.0)
up_mouse_img = jnp.clip(ndimage.rotate(mouse_img, 180, reshape=True), 0.0, 1.0)
cheese_img = plt.imread(os.path.join(assets_dir, "cheese.png"))
shock_img = plt.imread(os.path.join(assets_dir, "shock.png"))


class BaseTMaze(PymdpEnv):
    """
    Shared T-Maze implementation with configurable layout and observation structure.
    Description: A T-shaped maze where an agent must navigate to find a reward, with:
    - 4 (or 5) locations: centre, left arm, right arm, cue position (bottom), and optionally a middle location
    - 2 reward conditions: reward in left or right arm
    - 2 possible cues at the bottom arm that indicate which arm contains the reward
    """

    reward_probability: float = field(static=True)
    punishment_probability: float = field(static=True)
    cue_validity: float = field(static=True)
    reward_condition: float = field(static=True)
    dependent_outcomes: bool = field(static=True)
    cue_mode: str = field(static=True)
    connectivity: str = field(static=True)
    has_middle: bool = field(static=True)
    num_locations: int = field(static=True)
    location_obs_size: int = field(static=True)

    def __init__(
        self,
        reward_probability: float = 1.0,
        punishment_probability: float = 1.0,
        cue_validity: float = 0.95,
        reward_condition: Optional[int] = None,
        dependent_outcomes: bool = False,
        *,
        num_locations: int,
        cue_mode: str,
        connectivity: str,
        has_middle: bool,
        location_obs_size: Optional[int] = None,
    ) -> None:
        """
        Initialize a configurable T-Maze environment.
        Args:
            reward_probability: Probability of receiving reward when choosing the correct arm
            punishment_probability: Probability of receiving punishment when choosing the incorrect arm
            cue_validity: Probability that the cue correctly indicates the reward location
            reward_condition: If specified, fixes reward to left (0) or right (1) arm, otherwise random
            dependent_outcomes: If True, punishment occurs as a function of reward probability (i.e., if reward probability is 0.8, then 20% punishment). If False, punishment occurs with set probability (i.e., 20% no outcome and punishment will only occur in the other (non-rewarding) arm)
            num_locations: Number of locations in the maze (4 or 5)
            cue_mode: 'separate' for separate cue modality, 'embedded' for embedded cue observations
            connectivity: 'fully_connected' or 'adjacent' for location transitions
            has_middle: If True, includes a middle location between arms
            location_obs_size: Size of location observation modality (required for embedded cue mode)
        """
        self.reward_probability = reward_probability
        self.punishment_probability = punishment_probability
        self.cue_validity = cue_validity
        self.reward_condition = reward_condition
        self.dependent_outcomes = dependent_outcomes
        self.cue_mode = cue_mode
        self.connectivity = connectivity
        self.has_middle = has_middle
        self.num_locations = num_locations

        self._location_indices = {"centre": 0, "left": 1, "right": 2, "cue": 3}
        if self.has_middle:
            self._location_indices["middle"] = 4

        if self.cue_mode == "embedded":
            self._loc_obs_cued_left = self._location_indices["cue"]
            self._loc_obs_cued_right = self._location_indices["cue"] + 1
            if location_obs_size is None:
                location_obs_size = self._loc_obs_cued_right + 1
        else:
            if location_obs_size is None:
                location_obs_size = self.num_locations
        self.location_obs_size = location_obs_size

        if self.cue_mode == "embedded":
            self._loc_obs_to_location = list(range(self._location_indices["cue"] + 1))
            self._loc_obs_to_location.append(self._location_indices["cue"])
            self._cue_obs_from_loc_obs = [0] * self.location_obs_size
            self._cue_obs_from_loc_obs[self._loc_obs_cued_left] = 1
            self._cue_obs_from_loc_obs[self._loc_obs_cued_right] = 2
        else:
            self._loc_obs_to_location = list(range(self.location_obs_size))
            self._cue_obs_from_loc_obs = None

        A, A_dependencies = self.generate_A()
        B, B_dependencies = self.generate_B()
        D = self.generate_D()

        super().__init__(A=A, B=B, D=D, A_dependencies=A_dependencies, B_dependencies=B_dependencies)

    def _set_reward_outcome(self, A_reward: jnp.ndarray, loc: int, reward_condition: int) -> jnp.ndarray:
        if loc == (reward_condition + 1):
            A_reward = A_reward.at[1, loc, reward_condition].set(self.reward_probability)
            if self.dependent_outcomes:
                A_reward = A_reward.at[2, loc, reward_condition].set(1 - self.reward_probability)
            else:
                A_reward = A_reward.at[0, loc, reward_condition].set(1 - self.reward_probability)
        else:
            if self.dependent_outcomes:
                A_reward = A_reward.at[2, loc, reward_condition].set(self.reward_probability)
                A_reward = A_reward.at[1, loc, reward_condition].set(1 - self.reward_probability)
            else:
                A_reward = A_reward.at[2, loc, reward_condition].set(self.punishment_probability)
                A_reward = A_reward.at[0, loc, reward_condition].set(1 - self.punishment_probability)
        return A_reward

    def generate_A(self) -> tuple[list[jnp.ndarray], list[list[int]]]:
        if self.cue_mode == "separate":
            return self._generate_A_separate()
        if self.cue_mode == "embedded":
            return self._generate_A_embedded()
        raise ValueError(f"Unsupported cue_mode: {self.cue_mode}")

    def _generate_A_separate(self) -> tuple[list[jnp.ndarray], list[list[int]]]:
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
        A.append(jnp.eye(self.num_locations))
        A.append(jnp.zeros([3, self.num_locations, 2]))
        A.append(jnp.zeros([3, self.num_locations, 2]))

        A_dependencies = [[0], [0, 1], [0, 1]]
        middle_idx = self._location_indices.get("middle")

        for loc in range(self.num_locations):
            for reward_condition in range(2):
                if loc == self._location_indices["centre"]:
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)
                    A[2] = A[2].at[0, loc, reward_condition].set(1.0)
                elif loc == self._location_indices["cue"]:
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)
                    A[2] = A[2].at[reward_condition + 1, loc, reward_condition].set(self.cue_validity)
                    wrong_cue = (1 - reward_condition) + 1
                    A[2] = A[2].at[wrong_cue, loc, reward_condition].set(1 - self.cue_validity)
                elif middle_idx is not None and loc == middle_idx:
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)
                    A[2] = A[2].at[0, loc, reward_condition].set(1.0)
                else:
                    A[1] = self._set_reward_outcome(A[1], loc, reward_condition)
                    A[2] = A[2].at[0, loc, reward_condition].set(1.0)

        return A, A_dependencies

    def _generate_A_embedded(self) -> tuple[list[jnp.ndarray], list[list[int]]]:
        """
        Generate observation likelihood tensors.
        
        Returns two observation matrices:
        A[0]: Location observations with embedded cues (location_obs_size x5x2 matrix)
            - Maps true location to observed location with embedded cue information
            - [centre, left, right, cued left, cued right] x [centre, left, right, cue, middle] x [2 reward conditions]
        A[1]: Reward observations (3x5x2 matrix)
            - [no outcome, reward, punishment] x [5 locations] x [2 reward conditions]
        """
        A = []
        A.append(jnp.zeros([self.location_obs_size, self.num_locations, 2]))
        A.append(jnp.zeros([3, self.num_locations, 2]))

        A_dependencies = [[0, 1], [0, 1]]

        for loc in range(self.num_locations):
            for reward_condition in range(2):
                if loc == self._location_indices["centre"]:
                    A[0] = A[0].at[0, loc, reward_condition].set(1.0)
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)
                elif loc == self._location_indices["cue"]:
                    if reward_condition == 0:
                        A[0] = A[0].at[self._loc_obs_cued_left, loc, reward_condition].set(self.cue_validity)
                        A[0] = A[0].at[self._loc_obs_cued_right, loc, reward_condition].set(1 - self.cue_validity)
                    else:
                        A[0] = A[0].at[self._loc_obs_cued_right, loc, reward_condition].set(self.cue_validity)
                        A[0] = A[0].at[self._loc_obs_cued_left, loc, reward_condition].set(1 - self.cue_validity)
                    A[1] = A[1].at[0, loc, reward_condition].set(1.0)
                else:
                    A[0] = A[0].at[loc, loc, reward_condition].set(1.0)
                    A[1] = self._set_reward_outcome(A[1], loc, reward_condition)

        return A, A_dependencies

    def _valid_connections(self) -> list[tuple[int, int]]:
        centre = self._location_indices["centre"]
        left = self._location_indices["left"]
        right = self._location_indices["right"]
        cue = self._location_indices["cue"]

        if self.has_middle:
            middle = self._location_indices["middle"]
            return [
                (centre, cue),
                (cue, centre),
                (centre, middle),
                (middle, centre),
                (middle, left),
                (left, middle),
                (middle, right),
                (right, middle),
            ]
        return [
            (centre, cue),
            (cue, centre),
            (centre, left),
            (left, centre),
            (centre, right),
            (right, centre),
        ]

    def generate_B(self) -> tuple[list[jnp.ndarray], list[list[int]]]:
        """
        Generate transition model matrices.
        
        Returns two transition matrices:
        B[0]: Location transitions (num_locations x num_locations x num_locations)
            - Agent can move between either
             all locations in the T-Maze (fully-connected) or adjacent locations, including a middle location between the two arms, in the T-Maze (adjacent)
        B[1]: Reward condition transitions (2x2x1)
            - Reward location stays fixed
        """
        B = []
        B_loc = jnp.zeros((self.num_locations, self.num_locations, self.num_locations))

        if self.connectivity == "fully_connected":
            for action in range(self.num_locations):
                B_loc = B_loc.at[action, :, action].set(1.0)
        elif self.connectivity == "adjacent":
            valid_connections = set(self._valid_connections())
            for _from in range(self.num_locations):
                for action in range(self.num_locations):
                    _to = action
                    if (_from, _to) in valid_connections:
                        B_loc = B_loc.at[_to, _from, action].set(1.0)
                    else:
                        B_loc = B_loc.at[_from, _from, action].set(1.0)
        else:
            raise ValueError(f"Unsupported connectivity: {self.connectivity}")

        B.append(B_loc)
        B_reward = jnp.eye(2).reshape(2, 2, 1)
        B.append(B_reward)
        B_dependencies = [[0], [1]]
        return B, B_dependencies

    def generate_D(self) -> list[jnp.ndarray]:
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
        D_loc = jnp.zeros([self.num_locations])
        D_loc = D_loc.at[self._location_indices["centre"]].set(1.0)
        D.append(D_loc)

        if self.reward_condition is None:
            D_reward = jnp.ones(2) * 0.5
        else:
            D_reward = jnp.zeros(2)
            D_reward = D_reward.at[self.reward_condition].set(1.0)
        D.append(D_reward)
        return D

    def render(
        self, observations: list[jnp.ndarray], mode: str = "human", title: Optional[str] = None
    ) -> jnp.ndarray | None:
        batch_size = observations[0].shape[0]

        plt.clf()

       # create n x n subplots for the batch_size
        n = math.ceil(math.sqrt(batch_size))
        fig, axes = plt.subplots(n, n, figsize=(6, 6))
        if title:
            fig.suptitle(title, fontsize=9)

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

            loc_obs = int(observations[0][i, 0])
            reward_obs = int(observations[1][i, 0])
            if self.cue_mode == "embedded":
                cue_obs = self._cue_obs_from_loc_obs[loc_obs]
                loc = self._loc_obs_to_location[loc_obs]
            else:
                cue_obs = int(observations[2][i, 0])
                loc = loc_obs

            if cue_obs == 0:
                cue_color = "tab:gray"
                edge_color = "tab:gray"
            elif cue_obs == 1:
                cue_color = "tab:orange"
                edge_color = "tab:gray"
            else:
                cue_color = "tab:purple"
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

            coords = None
            if loc == 1:
                coords = (0.5, 0.5)
            elif loc == 2:
                coords = (2.5, 0.5)

            if coords is not None:
                if reward_obs == 1:
                    cheese_im = OffsetImage(cheese_img, zoom=0.025 / n)
                    ab_cheese = AnnotationBbox(cheese_im, coords, xycoords="data", frameon=False)
                    an_cheese = ax.add_artist(ab_cheese)
                    an_cheese.set_zorder(2)
                elif reward_obs == 2:
                    shock_im = OffsetImage(shock_img, zoom=0.1 / n)
                    ab_shock = AnnotationBbox(shock_im, coords, xycoords="data", frameon=False)
                    ab_shock = ax.add_artist(ab_shock)
                    ab_shock.set_zorder(2)

            if loc == 0:
                up_mouse_im = OffsetImage(up_mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(up_mouse_im, (1.5, 1.5), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)
            elif loc == 1:
                left_mouse_im = OffsetImage(left_mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(left_mouse_im, (0.75, 0.5), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)
            elif loc == 2:
                right_mouse_im = OffsetImage(right_mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(right_mouse_im, (2.25, 0.5), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)
            elif loc == 3:
                down_mouse_im = OffsetImage(mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(down_mouse_im, (1.5, 2.25), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)
            elif loc == 4:
                middle_mouse_im = OffsetImage(up_mouse_img, zoom=0.04 / n)
                ab_mouse = AnnotationBbox(middle_mouse_im, (1.5, 0.5), xycoords="data", frameon=False)
                ab_mouse = ax.add_artist(ab_mouse)
                ab_mouse.set_zorder(3)

        for i in range(batch_size, n * n):
            fig.delaxes(axes.flatten()[i])

        if title:
            fig.tight_layout(rect=[0, 0, 1, 0.93])
        else:
            plt.tight_layout()

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            img = fig2img(fig)
            plt.close(fig)
            return img


class TMaze(BaseTMaze):
    """
    Classic T-Maze with a middle connector, adjacent transitions, and a separate cue modality.
    """

    def __init__(
        self,
        reward_probability: float = 1.0,
        punishment_probability: float = 1.0,
        cue_validity: float = 0.95,
        reward_condition: Optional[int] = None,
        dependent_outcomes: bool = False,
    ) -> None:
        super().__init__(
            reward_probability=reward_probability,
            punishment_probability=punishment_probability,
            cue_validity=cue_validity,
            reward_condition=reward_condition,
            dependent_outcomes=dependent_outcomes,
            num_locations=5,
            cue_mode="separate",
            connectivity="adjacent",
            has_middle=True,
        )


class SimplifiedTMaze(BaseTMaze):
    """
    Fully connected T-Maze with embedded cues and no middle connector.
    """

    def __init__(
        self,
        reward_condition: Optional[int] = None,
        cue_validity: float = 0.95,
        reward_probability: float = 1.0,
        dependent_outcomes: bool = False,
        punishment_probability: float = 1.0,
    ) -> None:
        super().__init__(
            reward_probability=reward_probability,
            punishment_probability=punishment_probability,
            cue_validity=cue_validity,
            reward_condition=reward_condition,
            dependent_outcomes=dependent_outcomes,
            num_locations=4,
            cue_mode="embedded",
            connectivity="fully_connected",
            has_middle=False,
            location_obs_size=5,
        )
