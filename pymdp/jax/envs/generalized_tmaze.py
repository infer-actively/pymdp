from pymdp.jax.envs import PyMDPEnv
import numpy as np

import matplotlib.pyplot as plt
import io
import PIL.Image

import jax.numpy as jnp
from matplotlib.lines import Line2D


def position_to_index(position, shape):
    """
    Maps the position in the grid to a flat index
    Parameters
    ----------
    position
        Tuple of position (row, col)
    shape
        The shape of the grid (n_rows, n_cols)

    Returns
    ----------
    index
        A flattened index of position
    """
    return position[0] * shape[1] + position[1]


def index_to_position(index, shape):
    """
    Maps the flat index to a position coordinate in the grid
    Parameters
    ----------
    shape
        The shape of the grid (n_rows, n_cols)
    index
        A flattened index of position

    Returns
    ----------
    position
        Tuple of position (row, col)
    """
    return index // shape[1], index % shape[1]


def parse_maze(maze):
    """
    Parameters
    ----------
    maze
        a matrix representation of the environment
        where indices have particular meaning:
        0: Empty space
        1: The initial position of the agent
        2: Walls
        3 + i: Cue for reward i
        4 + i: Potential reward location i 1
        4 + i: Potential reward location i 2
    Returns
    ----------
    env_info
        a dictionary containing the environment information needed for
        constructing the agent/environment matrices and visualization
        purposes
    """

    maze = np.array(maze)
    rows, cols = maze.shape

    num_cues = int((np.max(maze) - 2) // 3)

    cue_positions = []
    reward_1_positions = []
    reward_2_positions = []
    for i in range(num_cues):
        cue_positions.append(tuple(np.argwhere(maze == 3 + 3 * i)[0]))
        reward_1_positions.append(tuple(np.argwhere(maze == 4 + 3 * i)[0]))
        reward_2_positions.append(tuple(np.argwhere(maze == 5 + 3 * i)[0]))

    # Initialize agent's starting position (can be customized if required)
    initial_position = tuple(np.argwhere(maze == 1)[0])

    # Actions: up, down, left, right
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Set reward location randomly
    reward_locations = np.random.choice([0, 1], size=num_cues)
    reward_indices = []
    no_reward_indices = []

    for i in range(num_cues):
        if reward_locations[i] == 0:
            reward_indices += position_to_index(
                reward_1_positions[i], maze.shape
            )
            no_reward_indices += position_to_index(
                reward_2_positions[i], maze.shape
            )
        else:
            reward_indices += position_to_index(
                reward_2_positions[i], maze.shape
            )
            no_reward_indices += position_to_index(
                reward_1_positions[i], maze.shape
            )

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


def generate_A(maze_info):
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
    position_likelihood = np.zeros((num_states, num_states))
    for i in range(num_states):
        # Agent can be certain about its position regardless of reward state
        position_likelihood[i, i] = 1

    cue_likelihoods = []
    for i in range(num_cues):
        # Cue observation likelihood, cue_position = (11, 5)
        # obs (nothing, left location, right location)
        # state: (current position, reward i position)
        cue_likelihood = np.zeros((3, num_states, 2))
        cue_likelihood[0, :, :] = 1  # Default: no info about reward

        cue_state_idx = position_to_index(cue_positions[i], maze.shape)
        reward_1_state_idx = position_to_index(
            reward_1_positions[i], maze.shape
        )
        reward_2_state_idx = position_to_index(
            reward_2_positions[i], maze.shape
        )

        cue_likelihood[:, cue_state_idx, 0] = [0, 1, 0]  # Reward in r1
        cue_likelihood[:, cue_state_idx, 1] = [0, 0, 1]  # Reward in r2
        cue_likelihoods.append(cue_likelihood)

    # Reward observation likelihood, r1 = (4, 7), r2 = (8, 7)
    reward_likelihoods = []

    for i in range(num_cues):
        # observation (nothing, no reward, reward)
        reward_likelihood = np.zeros((3, num_states, 2))
        reward_likelihood[0, :, :] = 1  # Default: no reward

        reward_1_state_idx = position_to_index(
            reward_1_positions[i], maze.shape
        )

        reward_2_state_idx = position_to_index(
            reward_2_positions[i], maze.shape
        )

        # Reward in (8,4) if reward state is 0
        reward_likelihood[:, reward_1_state_idx, 0] = [0, 1, 0]
        # Reward in (8,8) if reward state is 0
        reward_likelihood[:, reward_2_state_idx, 0] = [0, 0, 1]
        # Reward in (8,4) if reward state is 0
        reward_likelihood[:, reward_1_state_idx, 1] = [0, 0, 1]
        # Reward in (8,8) if reward state is 0
        reward_likelihood[:, reward_2_state_idx, 1] = [0, 1, 0]
        reward_likelihoods.append(reward_likelihood)

    combined_likelihood = np.empty(1 + 2 * num_cues, dtype=object)
    combined_likelihood[0] = position_likelihood
    for j, cue_likelihood in enumerate(cue_likelihoods):
        combined_likelihood[1 + j] = cue_likelihood
    for j, reward_likelihood in enumerate(reward_likelihoods):
        combined_likelihood[1 + num_cues + j] = reward_likelihood

    likelihood_dependencies = (
        [[0]]
        + [[0, 1 + i] for i in range(num_cues)]
        + [[0, 1 + i] for i in range(num_cues)]
    )

    return combined_likelihood, likelihood_dependencies


def generate_B(maze_info):
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

    P = np.zeros((num_states, num_actions), dtype=int)

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
                P[s, a] = s
            else:
                P[s, a] = position_to_index((ns_row, ns_col), maze.shape)

    B = np.zeros((num_states, num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            ns = P[s, a]
            B[ns, s, a] = 1

    assert np.all(np.logical_or(B == 0, B == 1))
    assert np.allclose(B.sum(axis=0), 1)

    reward_transitions = []
    for i in range(num_cues):
        reward_transition = np.eye(2).reshape(2, 2, 1)
        reward_transitions.append(reward_transition)

    combined_transition = np.empty(1 + num_cues, dtype=object)
    combined_transition[0] = B
    for i, reward_transition in enumerate(reward_transitions):
        combined_transition[1 + i] = reward_transition

    transition_dependencies = [[0]] + [[i + 1] for i in range(num_cues)]

    return combined_transition, transition_dependencies


def generate_D(maze_info):
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

    D[0] = np.zeros(cols * rows)
    # Position of the agent when starting the environment
    D[0][position_to_index(initial_position, maze.shape)] = 1

    # Cue state i.e. where is the reward
    for i in range(num_cues):
        r1 = reward_locations[i]
        D[1 + i] = np.zeros(2)
        D[1 + i][r1] = 1

    return D


def render(maze_info, env_state):
    """
    Plots and returns the rendered environment.
    Parameters
    ----------
    maze_info:
        info dict returned from `parse_maze` which contains the information
        about the reward locations, initial positions, etc.
    env_state:
        The environment state as a GeneralizedTMazeEnv instance
    Returns
    ----------
    image:
        A render of the environment.
    """
    maze = maze_info["maze"].copy()
    num_cues = maze_info["num_cues"]
    cue_positions = maze_info["cue_positions"]
    reward_1_positions = maze_info["reward_1_positions"]
    reward_2_positions = maze_info["reward_2_positions"]

    current_position = env_state.state[0]
    current_position = index_to_position(current_position, maze.shape)

    # Set all states not in [1] to be 0 (accessible state)
    mask = np.isin(maze, [2], invert=True)
    maze[mask] = 0

    plt.imshow(maze, cmap="gray_r", origin="lower")

    cmap = plt.get_cmap("tab10")
    plt.scatter(
        [ci[1] for ci in cue_positions],
        [ci[0] for ci in cue_positions],
        color=[cmap(i) for i in range(len(cue_positions))],
        s=200,
        alpha=0.5,
    )
    plt.scatter(
        [ci[1] for ci in cue_positions],
        [ci[0] for ci in cue_positions],
        color="black",
        s=50,
        label="Cue",
        marker="x",
    )

    plt.scatter(
        [ri[1] for ri in reward_1_positions],
        [ri[0] for ri in reward_1_positions],
        color=[cmap(i) for i in range(len(cue_positions))],
        s=200,
        alpha=0.5,
    )
    plt.scatter(
        [ri[1] for ri in reward_1_positions],
        [ri[0] for ri in reward_1_positions],
        marker="o",
        color="red",
        s=50,
        label="Positive",
    )
    plt.scatter(
        [ri[1] for ri in reward_2_positions],
        [ri[0] for ri in reward_2_positions],
        color=[cmap(i) for i in range(len(cue_positions))],
        s=200,
        alpha=0.5,
    )
    plt.scatter(
        [ri[1] for ri in reward_2_positions],
        [ri[0] for ri in reward_2_positions],
        marker="o",
        color="blue",
        s=50,
        label="Negative",
    )

    plt.scatter(
        current_position[1],
        current_position[0],
        c="tab:green",
        marker="s",
        s=100,
        label="Agent",
    )

    plt.title("Generalized T-Maze Environment")

    handles, labels = plt.gca().get_legend_handles_labels()
    for i in range(num_cues):
        if i == 0:
            label = "Reward set"
        else:
            label = f"Distractor {i} set"
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
    #plt.axis("off")
    plt.tight_layout()

    # Capture the current figure as an image
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)

    plt.show()

    return image


class GeneralizedTMazeEnv(PyMDPEnv):
    """
    Extended version of the T-Maze in which there are multiple cues and reward pairs
    similar to the original T-maze.
    """

    def __init__(self, env_info):
        A, A_dependencies = generate_A(env_info)
        B, B_dependencies = generate_B(env_info)
        print(B_dependencies)
        D = generate_D(env_info)
        params = {
            "A": [jnp.expand_dims(jnp.array(x), 0) for x in A],
            "B": [jnp.expand_dims(jnp.array(x), 0) for x in B],
            "D": [jnp.expand_dims(jnp.array(x), 0) for x in D],
        }
        dependencies = {"A": A_dependencies, "B": B_dependencies}

        PyMDPEnv.__init__(self, params, dependencies)
