from .env import PymdpEnv
from typing import Any
import numpy as np
import math
import jax.numpy as jnp

import matplotlib.pyplot as plt
from pymdp.utils import fig2img

from jax import random as jr
from jaxtyping import PRNGKeyArray
from matplotlib.lines import Line2D


def get_maze_matrix(small: bool = False) -> np.ndarray:

    """
    We create a matrix representation of the T-maze environment
    
    Matrix values:
    0: Grid cells the agent can move through
    1: Initial position of the agent
    2: Walls of the grid cells
    3, 6, 9: Cue locations (multiples of 3, where 3 is cue for reward set 1, 6 for set 2, etc.)
    4+: Potential reward locations:
        4-5: Reward set 1 locations
        7-8: Reward set 2 locations
        10-11: Reward set 3 locations
    
    Args:
        small: If True, creates a 3x3 maze with 2 reward sets
               If False, creates a 5x5 maze with 3 reward sets
    """

    if small:
        M = np.zeros((3, 3))  # 3x3 grid

        # Set the reward locations for sets 1 and 2
        M[0,0] = 4  # First location of reward set 1
        M[1,0] = 5  # Second location of reward set 1
        M[1,2] = 7  # First location of reward set 2
        M[0,2] = 8  # Second location of reward set 2

        # Set the cue locations
        M[2,0] = 3  # Cue for reward set 1 (3 = 3*1)
        M[2,2] = 6  # Cue for reward set 2 (6 = 3*2)

        # Set the initial position
        M[2,1] = 1
    else:
        M = np.zeros((5, 5))  # 5x5 grid

        # Set the reward locations for sets 1, 2, and 3
        M[0,0] = 4   # First location of reward set 1
        M[1,0] = 5   # Second location of reward set 1
        M[1,4] = 7   # First location of reward set 2
        M[0,4] = 8   # Second location of reward set 2
        M[4,1] = 10  # First location of reward set 3
        M[4,3] = 11  # Second location of reward set 3

        # Set the cue locations
        M[2,1] = 3   # Cue for reward set 1 (3 = 3*1)
        M[2,3] = 6   # Cue for reward set 2 (6 = 3*2)
        M[3,2] = 9   # Cue for reward set 3 (9 = 3*3)

        # Set the initial position
        M[2,2] = 1

    return M

def parse_maze(maze: np.ndarray, rng_key: PRNGKeyArray) -> dict[str, Any]:
    """
    Parses the maze matrix into a format needed for the environment and its visualization.
    
    Parameters
    ----------
    maze : array
        A matrix representation of the environment where values represent:
        0: Grid cells the agent can move through
        1: Initial position of the agent
        2: Walls of the grid cells 
        3, 6, 9: Cue locations (multiples of 3, where 3 is cue for reward set 1, 6 for set 2, etc.)
        4+: Potential reward locations:
            4-5: Reward set 1 locations
            7-8: Reward set 2 locations
            10-11: Reward set 3 locations
    
    rng_key : PRNGKeyArray
        Random key for determining which location in each set will be reward vs punishment
    
    Returns
    ----------
    env_info : dict
        Dictionary containing:
        - maze: The original maze matrix
        - actions: Possible movements [(up), (down), (left), (right)]
        - num_cues: Number of cue-reward sets
        - cue_positions: Coordinates of each cue
        - reward_indices: Flattened indices of true reward locations
        - no_reward_indices: Flattened indices of punishment locations
        - initial_position: Starting coordinates of agent
        - reward_1_positions: Coordinates of first potential reward location in each set
        - reward_2_positions: Coordinates of second potential reward location in each set
        - reward_locations: Binary array indicating which position (1 or 2) contains reward for each set
    """
    rows, cols = maze.shape

    # Calculate number of cue-reward sets based on highest value in maze
    # (max value - 2) // 3 gives us number of sets because:
    # Set 1: cue=3, rewards=4,5
    # Set 2: cue=6, rewards=7,8
    # Set 3: cue=9, rewards=10,11
    num_cues = int((jnp.max(maze) - 2) // 3)

    
    # Store coordinates for each set's cue and potential reward locations
    cue_positions = []
    reward_1_positions = []
    reward_2_positions = []
    for i in range(num_cues):
        # For set i:
        # Cue value = 3 + 3i (3,6,9)
        cue_positions.append(tuple(jnp.argwhere(maze == 3 + 3 * i)[0]))
        # First reward location value = 4 + 3i (4,7,10)
        reward_1_positions.append(tuple(jnp.argwhere(maze == 4 + 3 * i)[0]))
        # Second reward location value = 5 + 3i (5,8,11)
        reward_2_positions.append(tuple(jnp.argwhere(maze == 5 + 3 * i)[0]))

    # Get agent's starting position (value = 1 in maze)
    initial_position = tuple(jnp.argwhere(maze == 1)[0])

    # Define possible movements in (row, col) format
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Randomly assign which location in each set will be reward vs punishment
    # reward_locations[i] = 0 means first location is reward, 1 means second location is reward
        # Using uniform random numbers
    key, subkey = jr.split(rng_key)
    reward_locations = (jr.uniform(subkey, shape=(num_cues,)) > 0.5).astype(jnp.int32)
    # reward_locations = jr.choice(rng_key, 2, shape=(num_cues,))
    reward_indices = []
    no_reward_indices = []

    # Convert 2D coordinates to flattened indices for each set
    for i in range(num_cues):
        if reward_locations[i] == 0:
            # First location (4,7,10) is reward
            reward_indices += [jnp.ravel_multi_index(jnp.array(reward_1_positions[i]), maze.shape).item()]
            # Second location (5,8,11) is punishment
            no_reward_indices += [jnp.ravel_multi_index(jnp.array(reward_2_positions[i]), maze.shape).item()]
        else:
            # Second location is reward
            reward_indices += [jnp.ravel_multi_index(jnp.array(reward_2_positions[i]), maze.shape).item()]
            # First location is punishment
            no_reward_indices += [jnp.ravel_multi_index(jnp.array(reward_1_positions[i]), maze.shape).item()]

    return {
        "maze": maze,
        "actions": actions,
        "num_cues": num_cues,
        "cue_positions": cue_positions,
        "reward_indices": reward_indices,
        "no_reward_indices": no_reward_indices,
        "initial_position": initial_position,
        "reward_1_positions": reward_1_positions,
        "reward_2_positions": reward_2_positions,
        "reward_locations": reward_locations,
    }


def generate_A(maze_info: dict[str, Any]) -> tuple[list[jnp.ndarray], list[list[int]]]:
    """
    Parameters
    ----------
    maze_info:
        info dict returned from `parse_maze` which contains the information
        about the reward locations, initial positions, etc.
    Returns
    ----------
    A matrix:
        The likelihood mapping for the generalized T-maze. Maps the observations
        of (position, *cue_i, *reward_i) to states (position, reward)
    A dependencies:
        The state dependencies that generate observation for modality i
    """
    # Positional observation likelihood
    maze = maze_info["maze"]
    rows, cols = maze.shape
    num_cues = maze_info["num_cues"]
    cue_positions = maze_info["cue_positions"]
    reward_1_positions = maze_info["reward_1_positions"]
    reward_2_positions = maze_info["reward_2_positions"]

    num_states = rows * cols
    position_likelihood = jnp.zeros((num_states, num_states))
    for i in range(num_states):
        # Agent can be certain about its position regardless of reward state
        position_likelihood = position_likelihood.at[i, i].set(1)

    cue_likelihoods = []
    for i in range(num_cues):
        # Cue observation likelihood, cue_position = (11, 5)
        # obs (nothing, left location, right location)
        # state: (current position, reward i position)
        cue_likelihood = jnp.zeros((3, num_states, 2))
        cue_likelihood = cue_likelihood.at[0, :, :].set(1)  # Default: no info about reward

        cue_state_idx = jnp.ravel_multi_index(jnp.array(cue_positions[i]), maze.shape)
        reward_1_state_idx = jnp.ravel_multi_index(jnp.array(reward_1_positions[i]), maze.shape)
        reward_2_state_idx = jnp.ravel_multi_index(jnp.array(reward_2_positions[i]), maze.shape)

        cue_likelihood = cue_likelihood.at[:, cue_state_idx, 0].set(jnp.array([0, 1, 0]))  # Reward in r1
        cue_likelihood = cue_likelihood.at[:, cue_state_idx, 1].set(jnp.array([0, 0, 1]))  # Reward in r2
        cue_likelihoods.append(cue_likelihood)

    # Reward observation likelihood, r1 = (4, 7), r2 = (8, 7)
    reward_likelihoods = []

    for i in range(num_cues):
        # observation (nothing, no reward, reward)
        reward_likelihood = jnp.zeros((3, num_states, 2))
        reward_likelihood = reward_likelihood.at[0, :, :].set(1)  # Default: no reward

        reward_1_state_idx = jnp.ravel_multi_index(jnp.array(reward_1_positions[i]), maze.shape)
        reward_2_state_idx = jnp.ravel_multi_index(jnp.array(reward_2_positions[i]), maze.shape)

        # Reward in (8,4) if reward state is 0
        reward_likelihood = reward_likelihood.at[:, reward_1_state_idx, 0].set(jnp.array([0, 1, 0]))
        # Reward in (8,8) if reward state is 0
        reward_likelihood = reward_likelihood.at[:, reward_2_state_idx, 0].set(jnp.array([0, 0, 1]))
        # Reward in (8,4) if reward state is 0
        reward_likelihood = reward_likelihood.at[:, reward_1_state_idx, 1].set(jnp.array([0, 0, 1]))
        # Reward in (8,8) if reward state is 0
        reward_likelihood = reward_likelihood.at[:, reward_2_state_idx, 1].set(jnp.array([0, 1, 0]))
        reward_likelihoods.append(reward_likelihood)

    combined_likelihood = []
    combined_likelihood.append(position_likelihood)
    for cue_likelihood in cue_likelihoods:
        combined_likelihood.append(cue_likelihood)
    for reward_likelihood in reward_likelihoods:
        combined_likelihood.append(reward_likelihood)

    likelihood_dependencies = (
        [[0]]
        + [[0, 1 + i] for i in range(num_cues)]
        + [[0, 1 + i] for i in range(num_cues)]
    )

    return combined_likelihood, likelihood_dependencies


def generate_B(maze_info: dict[str, Any]) -> tuple[list[jnp.ndarray], list[list[int]]]:
    """
    Parameters
    ----------
    maze_info:
        info dict returned from `parse_maze` which contains the information
        about the reward locations, initial positions, etc.
    Returns
    ----------
    B matrix:
        The transition matrix for the generalized T-maze. The position state
        is transitioned according to the maze layout, for the other states
        the transition matrix is the identity.
    B dependencies:
        The state dependencies that generate transition for state i
    """

    maze = maze_info["maze"]
    actions = maze_info["actions"]
    num_cues = maze_info["num_cues"]

    rows, cols = maze.shape
    num_states = rows * cols
    num_actions = len(actions)

    P = jnp.zeros((num_states, num_actions), dtype=int)

    for s in range(num_states):
        row, col = divmod(s, cols)

        for a in range(num_actions):
            ns_row, ns_col = row + actions[a][0], col + actions[a][1]

            if (
                ns_row < 0
                or ns_row >= rows
                or ns_col < 0
                or ns_col >= cols
                or maze[ns_row, ns_col] == 2
            ):
                P = P.at[s, a].set(s)
            else:
                P = P.at[s, a].set(jnp.ravel_multi_index(jnp.array((ns_row, ns_col)), maze.shape))

    B = jnp.zeros((num_states, num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            ns = P[s, a]
            B = B.at[ns, s, a].set(1)
    # add do nothing action 
    B = jnp.concatenate([B, jnp.eye(num_states)[..., None]], -1)

    assert jnp.all(jnp.logical_or(B == 0, B == 1))
    assert jnp.allclose(B.sum(axis=0), 1)

    reward_transitions = []
    for i in range(num_cues):
        reward_transition = jnp.eye(2).reshape(2, 2, 1)
        reward_transitions.append(reward_transition)

    combined_transition = []
    combined_transition.append(B)
    for reward_transition in reward_transitions:
        combined_transition.append(reward_transition)

    transition_dependencies = [[0]] + [[i + 1] for i in range(num_cues)]

    return combined_transition, transition_dependencies


def generate_D(maze_info: dict[str, Any]) -> list[jnp.ndarray]:
    """
    Parameters
    ----------
    maze_info:
        info dict returned from `parse_maze` which contains the information
        about the reward locations, initial positions, etc.
    Returns
    ----------
    D vector:
        The initial state for the environment, i.e. each state is a one hot
        based on the environment initial conditions.
    """
    maze = maze_info["maze"]
    rows, cols = maze.shape
    num_cues = maze_info["num_cues"]
    reward_locations = maze_info["reward_locations"]
    initial_position = maze_info["initial_position"]

    D = [None for _ in range(1 + num_cues)]

    D[0] = jnp.zeros(cols * rows)
    # Position of the agent when starting the environment
    D[0] = D[0].at[jnp.ravel_multi_index(jnp.array(initial_position), maze.shape)].set(1)

    # Cue state i.e. where is the reward
    for i in range(num_cues):
        r1 = reward_locations[i]
        D[1 + i] = jnp.zeros(2)
        D[1 + i] = D[1 + i].at[r1].set(1)

    return D
class GeneralizedTMazeEnv(PymdpEnv):
    """
    Extended version of the T-Maze in which there are multiple cues and reward pairs
    similar to the original T-maze.
    """

    def __init__(self, env_info: dict[str, Any], categorical_obs: bool = False) -> None:
        """Initialize the generalized T-maze environment.

        Parameters
        ----------
        env_info : dict[str, Any]
            Environment specification used to construct the maze dynamics.
        categorical_obs : bool, default=False
            If ``True``, ``reset()`` and ``step()`` emit one-hot categorical
            observation vectors with shape ``(1, num_obs_m)`` for each
            modality. If ``False``, they emit discrete observation indices with
            shape ``(1,)``.
        """
        A, A_dependencies = generate_A(env_info)
        B, B_dependencies = generate_B(env_info)
        D = generate_D(env_info)
        super().__init__(
            A,
            B,
            D,
            A_dependencies,
            B_dependencies,
            categorical_obs=categorical_obs,
        )
        self.env_info = env_info

    def render(self, states: list[jnp.ndarray], mode: str = "human") -> jnp.ndarray | None:
        """
        Renders the environment
        Parameters
        ----------
        states:
            The environment states to render
        mode: str, optional
            The mode to render with ("human" or "rgb_array")
        Returns
        ----------
        if mode == "human":
            returns None, renders the environment using matplotlib inside the function
        elif mode == "rgb_array":
            A (H, W, 3) jax.numpy array that can act as input to functions like plt.imshow, with values between 0 and 255
        """

        if (states[0].ndim == 0):
            states = [jnp.expand_dims(s,0) for s in states]  # add batch dimension

        batch_size = states[0].shape[0]

        plt.clf()  # Clear the current figure

        maze = self.env_info["maze"].copy()
        num_cues = self.env_info["num_cues"]
        cue_positions = self.env_info["cue_positions"]
        reward_1_positions = self.env_info["reward_1_positions"]
        reward_2_positions = self.env_info["reward_2_positions"]

        # Set all states not in [1] to be 0 (accessible state)
        mask = np.isin(maze, [2], invert=True)
        maze[mask] = 0

        # create n x n subplots for the batch_size
        n = math.ceil(math.sqrt(batch_size))

        # create the subplots
        fig, axes = plt.subplots(n, n, figsize=(8,8),squeeze=False)
        axes_flat = axes.ravel()

        cmap = plt.get_cmap("tab10")

        for (i, ax) in enumerate(axes_flat[:batch_size]):
            
            current_position = states[0][i]
            current_position = jnp.unravel_index(current_position, maze.shape) # (row, col)

            ax.imshow(maze, cmap="gray_r", origin="lower")

            # cues
            ax.scatter(
                [ci[1] for ci in cue_positions],
                [ci[0] for ci in cue_positions],
                color=[cmap(i) for i in range(len(cue_positions))],
                s=200,
                alpha=0.5,
            )
            ax.scatter(
                [ci[1] for ci in cue_positions],
                [ci[0] for ci in cue_positions],
                color="black",
                s=50,
                label="Cue",
                marker="x",
            )

            # reward candidates
            ax.scatter(
                [ri[1] for ri in reward_1_positions],
                [ri[0] for ri in reward_1_positions],
                color=[cmap(i) for i in range(len(cue_positions))],
                s=200,
                alpha=0.5,
            )

            ax.scatter(
                [ri[1] for ri in reward_2_positions],
                [ri[0] for ri in reward_2_positions],
                color=[cmap(i) for i in range(len(cue_positions))],
                s=200,
                alpha=0.5,
            )

            for j, (r1, r2) in enumerate(zip(reward_1_positions, reward_2_positions)):
                if j == self.env_info["num_cues"] - 1:  # Only for the true reward set
                    if self.env_info["reward_locations"][j] == 0:
                        # First location is reward
                        ax.scatter(r1[1], r1[0], color='red', s=100, marker='o', zorder=4, label = "Reward")  # Red dot
                        ax.scatter(r2[1], r2[0], color='blue', s=100, marker='o', zorder=4, label = "Punishment")  # Blue dot
                    else:
                        # Second location is reward
                        ax.scatter(r2[1], r2[0], color='red', s=100, marker='o', zorder=4, label = "Reward")  # Red dot
                        ax.scatter(r1[1], r1[0], color='blue', s=100, marker='o', zorder=4, label = "Punishment")  # Blue dot


            ax.scatter(
                current_position[1],
                current_position[0],
                c="tab:green",
                marker="s",
                s=100,
                label="Agent",
            )

        # hide unused axes
        for k in range(batch_size, n * n):
            axes_flat[k].axis("off")

        plt.suptitle("Generalized T-Maze Environment")

        base_ax = axes_flat[next(i for i in range(len(axes_flat)) if axes_flat[i].has_data())]  # get  
        handles, labels = base_ax.get_legend_handles_labels()
        for i in range(num_cues):
            if i == num_cues - 1:
                label = "True Reward Set"
            else:
                label = f"Distractor Set {i + 1}"
            patch = Line2D(
                [0],
                [0],
                marker="o",
                markersize=10,
                markerfacecolor=cmap(i),
                markeredgecolor=cmap(i),
                label=label,
                alpha=0.5,
                linestyle="",
            )
            handles.append(patch)

        plt.legend(
            handles=handles, loc="upper left", bbox_to_anchor=(1, 1), fancybox=True
        )
        plt.tight_layout()

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            img = fig2img(fig)
            plt.close(fig) 
            return img
    
