import jax.numpy as jnp
import matplotlib.patches as patches
from pymdp.utils import fig2img
from equinox import field
from .env import Env
import matplotlib.pyplot as plt
from jax import nn
# from pymdp.learning import LearningConfig  # Not available in this branch
from typing import Dict, Any


class SimplestEnv(Env):
    """
    Implementation of the simplest environment in JAX. Useful for debugging and prototyping new features.
    This environment has two states (locations) and serves as a minimal test case for pymdp.
    Each state represents a location (left=0, right=1).
    There are two possible actions (left=0, right=1) which deterministically lead to their respective states.
    This is a fully observed environment: i.e., the observation likelihood matrix A is the identity matrix,
    meaning that each state maps deterministically to its corresponding observation with probability 1.
    This makes the true state of the environment directly observable to the agent.
    """

    state: jnp.ndarray = field(static=False)

    def __init__(self, batch_size=1):
        """
        Initialize the simplest environment.
        
        Args:
            batch_size: Number of parallel environments
        """
        # Generate and broadcast observation likelihood(A), transition (B), and initial state (D) tensors
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

        dependencies = {
            "A": A_dependencies,
            "B": B_dependencies,
        }

        # Pass parameters to parent class without labels (will use our overridden _initialize_default_labels method)
        super().__init__(params=params, dependencies=dependencies)

        
    def generate_A(self):
        """
        Generate observation likelihood tensor.
        Maps true location to observed location with a simple identity mapping.
        """
        A = [jnp.eye(2)] # Simple identity mapping between states and observations

        A_dependencies = [[0]]  # Only depends on location factor
        
        return A, A_dependencies

    def generate_B(self):
        """
        Generate transition tensor.
        Shape: (next_state, current_state, action)
        For each action (left=0, right=1), we deterministically transition to that state.
        
        B[0] has shape (2, 2, 2) where:
        - First dimension (2): next state (left=0, right=1)
        - Second dimension (2): current state (left=0, right=1)
        - Third dimension (2): action (left=0, right=1)
        
        For action=left (0):
            [[1, 1],  # Always go to left state (state 0)
             [0, 0]]
        For action=right (1):
            [[0, 0],  # Always go to right state (state 1)
             [1, 1]]
        """
        B = []
        
        # Initialize transition tensor
        B_locs = jnp.zeros((2, 2, 2))
        
        # For action 0 (left), always go to state 0 (left)
        B_locs = B_locs.at[0, :, 0].set(1.0)
        
        # For action 1 (right), always go to state 1 (right)
        B_locs = B_locs.at[1, :, 1].set(1.0)
        
        B.append(B_locs)
        
        B_dependencies = [[0]]  # Only depends on location factor
        
        return B, B_dependencies

    def generate_D(self):
        """
        Generate initial state distribution.
        Always starts at location 0 (left).
        
        This method serves two important roles:
        1. Environment Initialization: Determines the actual starting state distribution
           when the environment is reset. With [1.0, 0.0], the environment will always
           start deterministically in the left (0) state.
        2. Agent's Prior Beliefs: When an agent is created from this environment,
           this distribution becomes the agent's initial prior belief about states
           (unless explicitly overridden).
        
        Returns
        -------
        List[jnp.ndarray]
            A list containing the initial state distribution for the location factor.
            [1.0, 0.0] means deterministically start in state 0 (left).
            
        Notes
        -----
        We use a deterministic initial state ([1.0, 0.0]) rather than uniform ([0.5, 0.5])
        to make the environment behavior more predictable for testing and demonstration.
        This simplifies debugging.
        
        If you want to test D learning or have more randomized initial states, you could
        modify this to return a uniform distribution: jnp.array([0.5, 0.5]), 
        or use model.set_uniform_D() for the agent's generative model.
        """
        D = [jnp.array([1.0, 0.0])]  # Start at location 0 (left)
        return D

    def render(self, mode="human", observations=None):
        """
        Render the environment.
        
        Args:
            mode: 'human' displays plot, 'rgb_array' returns image array
            observations: List of observation arrays from environment
        
        Returns:
            Image array if mode=='rgb_array', else None
        """
        # Get batch size from observations or params
        if observations is not None:
            current_obs = observations
            batch_size = observations[0].shape[0]
        else:
            current_obs = self.current_obs
            batch_size = self.params["A"][0].shape[0]

        # Create figure with subplots (one per batch)
        n = int(jnp.ceil(jnp.sqrt(batch_size)))  # Square grid
        fig, axes = plt.subplots(n, n, figsize=(6, 6))

        # For each environment in the batch
        for i in range(batch_size):
            row = i // n
            col = i % n
            if batch_size == 1:
                ax = axes
            else:
                ax = axes[row, col]
            
            # Set up the plot
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Left', 'Right'])
            ax.set_yticks([])
            ax.set_aspect('equal')
            
            # Draw the two positions
            left_pos = patches.Circle((0, 0), 0.2, facecolor='lightgray', edgecolor='black')
            right_pos = patches.Circle((1, 0), 0.2, facecolor='lightgray', edgecolor='black')
            ax.add_patch(left_pos)
            ax.add_patch(right_pos)
            
            # Show agent position
            loc = current_obs[0][i, 0]
            agent_pos = patches.Circle((loc, 0), 0.1, facecolor='red')
            ax.add_patch(agent_pos)
            
            # Add arrow to show agent direction (this is just for aesthetics)
            ax.arrow(loc, 0, 0.15 if loc == 0 else -0.15, 0,
                    head_width=0.1, head_length=0.1, fc='red', ec='red')

        # Hide any extra subplots if batch_size isn't a perfect square
        for i in range(batch_size, n * n):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            img = fig2img(fig)
            plt.close(fig)
            return img


def plot_beliefs(info, agent=None, show=True):
    """Plot the agent's initial beliefs, final beliefs, and (if agent provided) preferences.
     
    Args:
        info: Rollout info dict with 'qs' for belief history
        agent: Agent instance (optional)
        show: Whether to call plt.show()
    """
    
    n_plots = 3 if agent is not None else 2
    plt.figure(figsize=(4 * n_plots, 4))

    # Plot initial beliefs as a bar plot
    plt.subplot(1, n_plots, 1)
    plt.bar([0, 1], info['empirical_prior'][0][0, 0, 0])  # (T+1, 1, 1, 2) -> get first timestep's beliefs
    plt.title('Initial Beliefs')
    plt.xticks([0, 1], ['Left', 'Right'])
    plt.ylim(0, 1)

    # Plot final beliefs as a bar plot
    plt.subplot(1, n_plots, 2)
    plt.bar([0, 1], info['qs'][0][-1, 0, 0])  # (T+1, 1, 1, 2) -> get last timestep's beliefs
    plt.title('Final Beliefs')
    plt.xticks([0, 1], ['Left', 'Right'])
    plt.ylim(0, 1)

    # Plot preferences as a bar plot
    if agent is not None:
        plt.subplot(1, n_plots, 3)
        plt.bar([0, 1], nn.softmax(agent.C[0][0]))
        plt.title('Preferences')
        plt.xticks([0, 1], ['Left', 'Right'])
        plt.ylim(0, 1)

    plt.tight_layout()
    if show:
        plt.show()
    
    return plt

def plot_A_learning(agent, info, env):
    """Plot the agent's learning progress for A matrix.
    
    Args:
        agent: Agent instance with parameter learning enabled
        info: Dict containing rollout info with parameter history
        env: Environment instance containing true parameters
    """
    
    plt.figure(figsize=(12, 5))
    plt.clf()  # Clear the current figure
    
    if agent.learn_A:
        # Create subplot for A matrix
        ax1 = plt.subplot(121)
        
        # Plot distance on left y-axis
        A_hist = info["agent"].A[0] # is the 0 indexing because of the batch-index? No it is the zeroth observation modality.
        timesteps = range(len(A_hist))
        distances = [jnp.linalg.norm(A - env.params["A"][0]) for A in A_hist]
        dist_line = ax1.plot(timesteps, distances, 'k--', label='Distance to true A', linewidth=2)[0]
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Distance to true parameters')
        ax1.set_ylim(bottom=0) # Set y-axis to include 0 
        
        # Create twin axis for probabilities
        ax2 = ax1.twinx()
        
        # Plot individual elements on right y-axis
        A_array = jnp.array(A_hist)
        lines = []
        lines.append(ax2.plot(timesteps, A_array[:, 0, 0], label='A[0,0]', alpha=0.5)[0])
        lines.append(ax2.plot(timesteps, A_array[:, 0, 1], label='A[0,1]', alpha=0.5)[0])
        lines.append(ax2.plot(timesteps, A_array[:, 1, 0], label='A[1,0]', alpha=0.5)[0])
        lines.append(ax2.plot(timesteps, A_array[:, 1, 1], label='A[1,1]', alpha=0.5)[0])
        ax2.set_ylabel('Belief')
        
        # Merge legends
        all_lines = [dist_line] + lines
        labs = [l.get_label() for l in all_lines]
        ax1.legend(all_lines, labs, loc='center left')
        
        plt.title('A Matrix Learning')
    
    plt.tight_layout()
    plt.show()
    
    return plt

def print_parameter_learning(info: Dict[str, Any], learning_config: Dict[str, bool]) -> None:
    """Print and analyze parameter learning results.
    
    Parameters
    ----------
    info : Dict[str, Any]
        Dictionary containing agent learning information
    learning_config : Dict[str, bool]
        Dictionary specifying which parameters are being learned.
        Expected keys: 'learn_A', 'learn_B', 'learn_D'
    """
    #TODO: If one passes action labels as arguments else use an index range, can reuse this function for multiple environments and put this in pymdp/analysis/learning.py 

    if learning_config['learn_A']:
        print('\n ====Parameter A learning====')
        print('\n Initial matrix A:\n', info["agent"].A[0][0,0,:])
        print('\n Final matrix A:\n', info["agent"].A[0][-1,0,:])

    if learning_config['learn_B']:
        print('\n ====Parameter B learning====')
        actions = ['Left', 'Right']
        for a in range(2): 
            print('\n Initial matrix B under action', actions[a], ':\n', info["agent"].B[0][0,0,:,:,a])
        for a in range(2): 
            print('\n Final matrix B under action', actions[a], ':\n', info["agent"].B[0][-1,0,:,:,a])

    if learning_config['learn_D']:
        print('\n ====Parameter D learning====')
        print('\n Initial D matrix:\n', info["agent"].D[0][0])
        print('\n Final learned D matrix:\n', info["agent"].D[0][-1])
        #TODO: add a verbose argument to print learned parameters at every timestep, such as below
        # for t in range(T+1):
        #     print(f't={t}, qD=', info["agent"].pD[0][t], 'D=', info["agent"].D[0][t])


def print_rollout(info, batch_idx=0):
    """Print a human-readable version of the rollout."""
    # Extract variables from info dictionary
    observations = info["observation"][0] # First modality, shape: (batch_size, T+1, 1)
    beliefs = info["qs"][0] # First factor, shape: (batch_size, T+1, 2)
    policies = info["qpi"] # Shape: (batch_size, T+1, num_policies)
    actions = info["action"] # Shape: (batch_size, T+1, 1)
    empirical_priors = info["empirical_prior"][0] # Shape: (batch_size, T+1, 2)
    
    # Get dimensions
    num_timesteps = observations.shape[1]
    
    # Print experiment setup
    print("\n=== Experiment Setup ===")
    print(f"Number of timesteps: {observations.shape[1]-1}")  # -1 because includes initial observation
    print(f"Batch size: {observations.shape[0]}")
    print(f"Number of policies: {policies.shape[-1]}")
    
    def format_state_dist(left_prob, right_prob):
        """Helper to format state distribution nicely"""
        return f"[L: {float(left_prob):.3f}, R: {float(right_prob):.3f}]"
    
    # Print initial timestep info
    print("\n=== Initial Timestep (t=0) ===")
    print("Prior beliefs (D):", format_state_dist(empirical_priors[batch_idx, 0, 0], empirical_priors[batch_idx, 0, 1]))
    print(f"Observation: [{['Left', 'Right'][int(observations[batch_idx, 0, 0])]}]")
    print("Posterior beliefs:", format_state_dist(beliefs[batch_idx, 0, 0], beliefs[batch_idx, 0, 1]))
    print("-" * 50)

    # Print trajectory
    for t in range(1, num_timesteps):
        print(f"\n=== Timestep {t} ===")
        
        # Print policy distribution
        print("Policy selection:")
        for p_idx, p_prob in enumerate(policies[batch_idx, t]):
            prob_str = f"{float(p_prob):.3f}"
            print(f"  Policy {p_idx:<15} : {prob_str:>8}")
 
        # Print action and its consequences
        next_action = int(actions[batch_idx, t, 0].item())
        print(f"Action: [Move to {['Left', 'Right'][next_action]}]")
        
        # Print prediction (empirical prior)
        print("Predicted next state:", format_state_dist(
            empirical_priors[batch_idx, t, 0],
            empirical_priors[batch_idx, t, 1]
        ))
        
        # Print actual observation and posterior
        next_obs = int(observations[batch_idx, t, 0].item())
        print(f"Observation: [{['Left', 'Right'][next_obs]}]")
        print("Posterior beliefs:", format_state_dist(beliefs[batch_idx, t, 0], beliefs[batch_idx, t, 1]))
        print("-" * 50)
    
    print("\n=== End of Experiment ===")
