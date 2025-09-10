import jax.numpy as jnp
import matplotlib.patches as patches
from pymdp.utils import fig2img
from equinox import field
from .env import Env
import matplotlib.pyplot as plt
from jax import nn
# from pymdp.learning import LearningConfig  # Not available in this branch
from typing import Dict, Any, List, Tuple, Optional, Union


class SimplestEnv(Env):
    """
    The simplest possible Active Inference environment for testing and learning.
    A minimal two-location world with:
    - 2 states: Left (0) and Right (1) locations  
    - 2 observations: Left (0) and Right (1)
    - 2 actions: Go Left (0) and Go Right (1)
    - Fully observable (i.e. A = identity matrix)
    - Deterministic transitions (actions always succeed)
    """

    state: jnp.ndarray = field(static=False)

    def __init__(self, batch_size: int = 1) -> None:
        """
        Initialize SimplestEnv.
        
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

        
    def generate_A(self) -> Tuple[List[jnp.ndarray], List[List[int]]]:
        """
        Generate observation likelihood tensor.
        
        Returns identity matrix for perfect observability:
        A[0]: Location observations (2x2 identity matrix)
            - Maps true location to observed location
            - [Left, Right] x [Left, Right]
        """
        A = [jnp.eye(2)]  # Identity mapping: perfect observability
        A_dependencies = [[0]]  # Location observation depends on location state factor
        
        return A, A_dependencies

    def generate_B(self) -> Tuple[List[jnp.ndarray], List[List[int]]]:
        """
        Generate transition tensor.
        
        Deterministic transitions where actions always succeed:
        B[0]: Location transitions (2x2x2 tensor)
            - Shape: (next_state, current_state, action)
            - Action 0 (Left): Always go to state 0 regardless of current state
            - Action 1 (Right): Always go to state 1 regardless of current state
        """
        B = []
        
        # Initialize transition tensor: B[next_state, current_state, action]
        B_locs = jnp.zeros((2, 2, 2))
        
        # Action 0 (Go Left): Always transition to state 0 regardless of current state
        B_locs = B_locs.at[0, :, 0].set(1.0)
        
        # Action 1 (Go Right): Always transition to state 1 regardless of current state
        B_locs = B_locs.at[1, :, 1].set(1.0)
        
        B.append(B_locs)
        B_dependencies = [[0]]  # Location state depends on movement control factor
        
        return B, B_dependencies

    def generate_D(self) -> List[jnp.ndarray]:
        """
        Generate initial state distribution.
        
        Always starts at Left location (state 0) for predictable behavior.
        D[0]: Initial location distribution [1.0, 0.0]
        """
        D = [jnp.array([1.0, 0.0])]  # Deterministically start at Left location
        return D

    def render(self, mode: str = "human", observations: Optional[List[jnp.ndarray]] = None) -> Optional[jnp.ndarray]:
        """
        Render the environment state.
        
        Shows agent position as red circle on a 2D grid with Left and Right locations.
        
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


def plot_beliefs(info: Dict[str, Any], agent: Optional[Any] = None, 
                 show: bool = True, batch_idx: int = 0) -> plt:
    """
    Plot agent's initial beliefs, final beliefs, and (if agent provided) preferences.
     
    Args:
        info: Rollout info dict with 'qs' for belief history
        agent: Agent instance (optional)
        show: Whether to call plt.show()
        batch_idx: Which batch element to plot (default: 0)
    """
    
    n_plots = 3 if agent is not None else 2
    plt.figure(figsize=(4 * n_plots, 4))

    # Plot initial beliefs as a bar plot
    plt.subplot(1, n_plots, 1)
    plt.bar([0, 1], info['empirical_prior'][0][batch_idx, 0])  # (batch_size, T+1, 2) -> get first timestep's beliefs
    plt.title('Initial Beliefs')
    plt.xticks([0, 1], ['Left', 'Right'])
    plt.ylim(0, 1)

    # Plot final beliefs as a bar plot
    plt.subplot(1, n_plots, 2)
    plt.bar([0, 1], info['qs'][0][batch_idx, -1])  # (batch_size, T+1, 2) -> get last timestep's beliefs
    plt.title('Final Beliefs')
    plt.xticks([0, 1], ['Left', 'Right'])
    plt.ylim(0, 1)

    # Plot preferences as a bar plot
    if agent is not None:
        plt.subplot(1, n_plots, 3)
        plt.bar([0, 1], nn.softmax(agent.C[0][batch_idx]))
        plt.title('Preferences')
        plt.xticks([0, 1], ['Left', 'Right'])
        plt.ylim(0, 1)

    plt.tight_layout()
    if show:
        plt.show()
    
    return plt

# Legacy function:
# def plot_A_learning(agent, info, env):
#     """Plot the agent's learning progress for A matrix.
    
#     Args:
#         agent: Agent instance with parameter learning enabled
#         info: Dict containing rollout info with parameter history
#         env: Environment instance containing true parameters
#     """
    
#     plt.figure(figsize=(12, 5))
#     plt.clf()  # Clear the current figure
    
#     if agent.learn_A:
#         # Create subplot for A matrix
#         ax1 = plt.subplot(121)
        
#         # Plot distance on left y-axis
#         A_hist = info["agent"].A[0] # is the 0 indexing because of the batch-index? No it is the zeroth observation modality.
#         timesteps = range(len(A_hist))
#         distances = [jnp.linalg.norm(A - env.params["A"][0]) for A in A_hist]
#         dist_line = ax1.plot(timesteps, distances, 'k--', label='Distance to true A', linewidth=2)[0]
#         ax1.set_xlabel('Timestep')
#         ax1.set_ylabel('Distance to true parameters')
#         ax1.set_ylim(bottom=0) # Set y-axis to include 0 
        
#         # Create twin axis for probabilities
#         ax2 = ax1.twinx()
        
#         # Plot individual elements on right y-axis
#         A_array = jnp.array(A_hist)
#         lines = []
#         lines.append(ax2.plot(timesteps, A_array[:, 0, 0], label='A[0,0]', alpha=0.5)[0])
#         lines.append(ax2.plot(timesteps, A_array[:, 0, 1], label='A[0,1]', alpha=0.5)[0])
#         lines.append(ax2.plot(timesteps, A_array[:, 1, 0], label='A[1,0]', alpha=0.5)[0])
#         lines.append(ax2.plot(timesteps, A_array[:, 1, 1], label='A[1,1]', alpha=0.5)[0])
#         ax2.set_ylabel('Belief')
        
#         # Merge legends
#         all_lines = [dist_line] + lines
#         labs = [l.get_label() for l in all_lines]
#         ax1.legend(all_lines, labs, loc='center left')
        
#         plt.title('A Matrix Learning')
    
#     plt.tight_layout()
#     plt.show()
    
#     return plt

# def print_parameter_learning(info: Dict[str, Any], learning_config: Dict[str, bool]) -> None:
#     """Print and analyze parameter learning results.
    
#     Parameters
#     ----------
#     info : Dict[str, Any]
#         Dictionary containing agent learning information
#     learning_config : Dict[str, bool]
#         Dictionary specifying which parameters are being learned.
#         Expected keys: 'learn_A', 'learn_B', 'learn_D'
#     """
#     #TODO: If one passes action labels as arguments else use an index range, can reuse this function for multiple environments and put this in pymdp/analysis/learning.py 

#     if learning_config['learn_A']:
#         print('\n ====Parameter A learning====')
#         print('\n Initial matrix A:\n', info["agent"].A[0][0,0,:])
#         print('\n Final matrix A:\n', info["agent"].A[0][-1,0,:])

#     if learning_config['learn_B']:
#         print('\n ====Parameter B learning====')
#         actions = ['Left', 'Right']
#         for a in range(2): 
#             print('\n Initial matrix B under action', actions[a], ':\n', info["agent"].B[0][0,0,:,:,a])
#         for a in range(2): 
#             print('\n Final matrix B under action', actions[a], ':\n', info["agent"].B[0][-1,0,:,:,a])

#     if learning_config['learn_D']:
#         print('\n ====Parameter D learning====')
#         print('\n Initial D matrix:\n', info["agent"].D[0][0])
#         print('\n Final learned D matrix:\n', info["agent"].D[0][-1])
#         #TODO: add a verbose argument to print learned parameters at every timestep, such as below
#         # for t in range(T+1):
#         #     print(f't={t}, qD=', info["agent"].pD[0][t], 'D=', info["agent"].D[0][t])


def render_rollout(env, info, save_gif=False, filename=None, fps=1):
    """Render a video of the agent's trajectory through any environment that implements a render method.
    
    This function iterates through the rollout information and renders each timestep using the 
    environment's built-in render method. It works with any environment that implements the
    standard render(mode="rgb_array", observations=observations_t) interface.
    
    Parameters
    ----------
    env : Env
        Environment instance (TMaze, SimplestEnv, etc.)
    info : dict
        Dictionary containing rollout information, as returned by the rollout function
    save_gif : bool, optional
        Whether to save the animation as a gif, by default False
    filename : str, optional
        Path to save the gif if save_gif is True, by default None
    fps : int, optional
        Frames per second for the rendered video, by default 1
        
    Returns
    -------
    None
        Displays the animation in the notebook or saves it as a gif
    """
    import mediapy
    from PIL import Image
    import os
    from warnings import warn

    # Check if multi-trial
    try:
        from pymdp.envs.rollout import is_multi_trial
        is_multi, _ = is_multi_trial(info)
        if is_multi: 
            warn("render_rollout is currently not implemented for multi-trial rollouts.")
            return 0
    except ImportError:
        # If we can't import is_multi_trial, assume single trial
        pass
    
    # Get the number of timesteps in the rollout
    num_timesteps = info["observation"][0].shape[1]  # Shape: (batch_size, T+1, 1)
    
    # Get the number of observation modalities
    num_modalities = len(info["observation"])
    
    frames = [None] * num_timesteps
    for t in range(num_timesteps):  # iterate over timesteps
        # Prepare observations for current timestep
        observations_t = [info["observation"][mod_idx][:, t] for mod_idx in range(num_modalities)]
        
        # Call the environment's render method
        frame = env.render(mode="rgb_array", observations=observations_t)
        frames[t] = jnp.asarray(frame, dtype=jnp.uint8)
        plt.close()  # close the figure to prevent memory leak
    
    # Convert frames to array and display video
    frames = jnp.array(frames, dtype=jnp.uint8)
    mediapy.show_video(frames, fps=fps)
    
    # Save as gif if requested
    if save_gif:
        if filename is None:
            raise ValueError("If save_gif is True, a filename must be provided")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pil_frames = [Image.fromarray(frame) for frame in frames]
        pil_frames[0].save(
            filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(1000/fps),  # milliseconds per frame
            loop=0
        )


def print_rollout(info, batch_idx=0):
    """Print a human-readable version of the rollout for some batch index."""
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
