from pymdp.jax.envs import PyMDPEnv
import numpy as np

import matplotlib.pyplot as plt
import io
import PIL.Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import jax.numpy as jnp


def add_icon(ax, coord, img_path, zoom=0.1):
    """Helper function to add an icon at a specified coordinate."""
    image = plt.imread(img_path)
    im = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(
        im, (coord[1], coord[0]), frameon=False, box_alignment=(0.5, 0.5)
    )
    ax.add_artist(ab)


class GeneralizedTMaze:

    def __init__(self, M):
        # Maze representation based on input matrix M
        self.maze = np.array(M)
        self.rows, self.cols = self.maze.shape

        self.num_cues = int((np.max(M) - 2) // 3)

        self.cue_positions = []
        self.reward_1_positions = []
        self.reward_2_positions = []
        for i in range(self.num_cues):
            self.cue_positions.append(
                tuple(np.argwhere(self.maze == 3 + 3 * i)[0])
            )
            self.reward_1_positions.append(
                tuple(np.argwhere(self.maze == 4 + 3 * i)[0])
            )
            self.reward_2_positions.append(
                tuple(np.argwhere(self.maze == 5 + 3 * i)[0])
            )

        # Initialize agent's starting position (can be customized if required)
        self.initial_position = tuple(np.argwhere(self.maze == 1)[0])
        self.current_position = self.initial_position

        # Actions: up, down, left, right
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Set reward locations
        self.reward_locations = np.random.choice([0, 1], size=self.num_cues)
        self.reward_indices = []
        self.no_reward_indices = []

        for i in range(self.num_cues):
            if self.reward_locations[i] == 0:
                self.reward_indices += [
                    self.reward_1_positions[i][0] * self.cols
                    + self.reward_1_positions[i][1]
                ]
                self.no_reward_indices += [
                    self.reward_2_positions[i][0] * self.cols
                    + self.reward_2_positions[i][1]
                ]
            else:
                self.reward_indices += [
                    self.reward_2_positions[i][0] * self.cols
                    + self.reward_2_positions[i][1]
                ]
                self.no_reward_indices += [
                    self.reward_1_positions[i][0] * self.cols
                    + self.reward_1_positions[i][1]
                ]

    def position_to_index(self, position):
        return position[0] * self.cols + position[1]

    def index_to_position(self, index):
        return index // self.cols, index % self.cols

    def compute_transition_matrix(self):
        num_states = self.rows * self.cols
        num_actions = 4

        P = np.zeros((num_states, num_actions), dtype=int)

        for s in range(num_states):
            row, col = divmod(s, self.cols)

            for a in range(num_actions):
                ns_row, ns_col = (
                    row + self.actions[a][0],
                    col + self.actions[a][1],
                )

                if (
                    ns_row < 0
                    or ns_row >= self.rows
                    or ns_col < 0
                    or ns_col >= self.cols
                    or self.maze[ns_row, ns_col] == 1
                ):
                    P[s, a] = s
                else:
                    P[s, a] = ns_row * self.cols + ns_col

        B = np.zeros((num_states, num_states, num_actions))
        for s in range(num_states):
            for a in range(num_actions):
                ns = P[s, a]
                B[ns, s, a] = 1

        assert np.all(np.logical_or(B == 0, B == 1))
        assert np.allclose(B.sum(axis=0), 1)

        reward_transitions = []
        for i in range(self.num_cues):
            reward_transition = np.eye(2).reshape(2, 2, 1)
            reward_transitions.append(reward_transition)

        combined_transition = np.empty(1 + self.num_cues, dtype=object)
        combined_transition[0] = B
        for i, reward_transition in enumerate(reward_transitions):
            combined_transition[1 + i] = reward_transition

        return combined_transition

    def compute_observation_likelihood(self):
        # Positional observation likelihood

        num_states = self.rows * self.cols
        position_likelihood = np.zeros((num_states, num_states))
        for i in range(num_states):
            # Agent can be certain about its position regardless of reward state
            position_likelihood[i, i] = 1

        cue_likelihoods = []
        for i in range(self.num_cues):
            # Cue observation likelihood, cue_position = (11, 5)
            # obs (nothing, left location, right location)
            # state: (current position, reward i position)
            cue_likelihood = np.zeros((3, num_states, 2))
            cue_likelihood[0, :, :] = 1  # Default: no info about reward

            cue_state_idx = self.position_to_index(self.cue_positions[i])
            reward_1_state_idx = self.position_to_index(
                self.reward_1_positions[i]
            )
            reward_2_state_idx = self.position_to_index(
                self.reward_2_positions[i]
            )

            cue_likelihood[:, cue_state_idx, 0] = [0, 1, 0]  # Reward in r1
            cue_likelihood[:, cue_state_idx, 1] = [0, 0, 1]  # Reward in r2
            cue_likelihoods.append(cue_likelihood)

        # Reward observation likelihood, r1 = (4, 7), r2 = (8, 7)
        reward_likelihoods = []

        for i in range(self.num_cues):
            # observation (nothing, no reward, reward)
            reward_likelihood = np.zeros((3, num_states, 2))
            reward_likelihood[0, :, :] = 1  # Default: no reward

            reward_1_state_idx = (
                self.reward_1_positions[i][0] * self.cols
                + self.reward_1_positions[i][1]
            )
            reward_2_state_idx = (
                self.reward_2_positions[i][0] * self.cols
                + self.reward_2_positions[i][1]
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

        combined_likelihood = np.empty(1 + 2 * self.num_cues, dtype=object)
        combined_likelihood[0] = position_likelihood
        for j, cue_likelihood in enumerate(cue_likelihoods):
            combined_likelihood[1 + j] = cue_likelihood
        for j, reward_likelihood in enumerate(reward_likelihoods):
            combined_likelihood[1 + self.num_cues + j] = reward_likelihood

        return combined_likelihood

    def compute_exact_D_vector(self):
        """
        Computes the prior over state, expecting perfect knowledge
        """
        D = [None for _ in range(1 + self.num_cues)]

        D[0] = np.zeros(self.cols * self.rows)
        # Position of the agent when starting the environment
        idx = self.position_to_index(self.initial_position)
        D[0][idx] = 1

        # Cue state i.e. where is the reward
        for i in range(self.num_cues):
            r1 = self.reward_locations[i]
            D[1 + i] = np.zeros(2)
            D[1 + i][r1] = 1

        return D

    def get_special_states_indices(self):
        """Return the indices of the cue state, reward_1 state, and reward_2 state from matrix M."""

        cue_idx = np.argwhere(self.maze == 4)[0]
        reward_1_idx = np.argwhere(self.maze == 2)[0]
        reward_2_idx = np.argwhere(self.maze == 3)[0]

        rows, cols = self.maze.shape
        cue_linear_idx = cue_idx[0] * cols + cue_idx[1]
        reward_1_linear_idx = reward_1_idx[0] * cols + reward_1_idx[1]
        reward_2_linear_idx = reward_2_idx[0] * cols + reward_2_idx[1]

        return cue_linear_idx, reward_1_linear_idx, reward_2_linear_idx

    def render_env(self, env_state):
        """
        Render the environment provided that the env state from the PyMDP equivalent
        is provided
        """
        current_position = env_state.state[0]
        current_position = self.index_to_position(current_position)

        # Create a copy of the maze for rendering
        maze_copy = np.copy(self.maze)

        # Set all states not in [1] to be 0 (accessible state)
        mask = np.isin(maze_copy, [2], invert=True)
        maze_copy[mask] = 0

        plt.imshow(maze_copy, cmap="gray_r", origin="lower")
        plt.scatter(
            current_position[1],
            current_position[0],
            color="green",
            marker="s",
            s=100,
            label="Agent",
        )

        c = plt.get_cmap("tab20")(0)

        plt.scatter(
            self.cue_positions[0][1],
            self.cue_positions[0][0],
            color=c,
            s=200,
            label="Reward of Interest",
        )
        plt.scatter(
            self.cue_positions[0][1],
            self.cue_positions[0][0],
            marker="o",
            color="blue",
            s=50,
            label="Cue",
        )
        plt.scatter(
            self.reward_1_positions[0][1],
            self.reward_1_positions[0][0],
            color=c,
            s=200,
        )
        plt.scatter(
            self.reward_1_positions[0][1],
            self.reward_1_positions[0][0],
            marker="o",
            color="red",
            s=50,
            label="Reward 1",
        )
        plt.scatter(
            self.reward_2_positions[0][1],
            self.reward_2_positions[0][0],
            color=c,
            s=200,
        )
        plt.scatter(
            self.reward_2_positions[0][1],
            self.reward_2_positions[0][0],
            marker="o",
            color="red",
            s=50,
            label="Reward 2",
        )

        for i in range(1, self.num_cues):
            c = plt.get_cmap("tab20")(i)
            plt.scatter(
                self.cue_positions[i][1],
                self.cue_positions[i][0],
                color=c,
                s=200,
            )
            plt.scatter(
                self.cue_positions[i][1],
                self.cue_positions[i][0],
                color="blue",
                s=50,
            )
            plt.scatter(
                self.reward_1_positions[i][1],
                self.reward_1_positions[i][0],
                color=c,
                s=200,
            )
            plt.scatter(
                self.reward_1_positions[i][1],
                self.reward_1_positions[i][0],
                color="red",
                s=50,
            )
            plt.scatter(
                self.reward_2_positions[i][1],
                self.reward_2_positions[i][0],
                color=c,
                s=200,
            )
            plt.scatter(
                self.reward_2_positions[i][1],
                self.reward_2_positions[i][0],
                color="red",
                s=50,
            )

        plt.title("Generalized T-Maze Environment")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Capture the current figure as an image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        image = PIL.Image.open(buf)

        plt.grid("on")
        plt.show()

        return image


class GeneralizedTMazeEnv(PyMDPEnv):
    """
    Extended version of the T-Maze in which there are multiple cues and reward pairs
    similar to the original T-maze.
    """

    def __init__(self, environment_description):
        """
        Parameters
        ----------
        environment_description
            The environment description is a matrix representation of the environment
            where indices have particular meaning:
            0: Empty space
            1: The initial position of the agent
            2: Walls
            3 + i: Cue for reward i
            4 + i: Potential reward location i 1
            4 + i: Potential reward location i 2
        """

        env = GeneralizedTMaze(environment_description)
        A = [
            jnp.expand_dims(jnp.array(a), 0)
            for a in env.compute_observation_likelihood()
        ]
        B = [
            jnp.expand_dims(jnp.array(b), 0)
            for b in env.compute_transition_matrix()
        ]
        D = [
            jnp.expand_dims(jnp.array(d), 0)
            for d in env.compute_exact_D_vector()
        ]

        params = {"A": A, "B": B, "D": D}
        dependencies = {
            "A": [[0]]
            + [[0, 1 + i] for i in range(len(D) - 1)]
            + [[0, 1 + i] for i in range(len(D) - 1)],
            "B": [[0]] + [[i + 1] for i in range(len(D) - 1)],
        }

        PyMDPEnv.__init__(self, params, dependencies)
