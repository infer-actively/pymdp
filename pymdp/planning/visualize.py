

from types import NoneType
from typing import Any, Callable
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu

import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation

from IPython.display import display, HTML


# default function to convert actions to string representation
# can be overridden by passing a custom function to the `plot_tree` function
def action_to_string(action: Any, model: Any = None) -> str:
    if model is None:
        return str(action)
    
    # group by control type to avoid duplicates
    control_actions = {}
    
    for i in range(len(model.B)):
        if i < len(action):
            action_idx = action[i]
            k = list(model.B[i].batch.keys())[-1]  # the control key
            actions_list = model.B[i].batch[k]
            
            if action_idx < len(actions_list):
                action_name = actions_list[action_idx]
                # Group by control type (k)
                if k not in control_actions:
                    control_actions[k] = action_name
    
    # Join all control types
    action_parts = [action_name for action_name in control_actions.values() if action_name != "noop"]
    
    if not action_parts:
        return "noop"
    
    return ", ".join(action_parts)


def observation_to_string(observation: Any, model: Any = None) -> str:
    if model is None:
        return str(observation[0])
    
    obs = ""
    for i in range(len(model.A)):
        for k,v in model.A[i].event.items():
            # v should have string values for each observation modality
            obs_str = str(v[observation[0, i]])
            obs += obs_str
            obs += ",\n"

    return obs


def formatting_jax(value: Any, format_str: str = ".2f") -> str:
    try:
        if hasattr(value, "shape"):
            if value.shape == ():
                return f"{float(value):{format_str}}"
            elif len(value.shape) == 1 and value.shape[0] == 1:
                return f"{float(value[0]):{format_str}}"
            elif len(value.shape) == 1 and value.shape[0] <= 4:
                return (
                    "[" + ", ".join([f"{float(x):{format_str}}" for x in value]) + "]"
                )
            else:
                return (
                    "[" + ", ".join([f"{float(x):{format_str}}" for x in value]) + "]"
                )
        else:
            return f"{float(value):{format_str}}"
    except Exception:
        return str(value)[:10]


def plot_plan_tree(
    tree: Any,
    model: Any = None,
    root_node: Any = None,
    max_depth: int = 4,
    min_prob: float = 0.2,
    observation_description: Callable = observation_to_string,
    action_description: Callable = action_to_string,
    figsize: tuple[int, int] = (8, 7),
    font_size: int = 10,
    node_size: int = 1500,
    layout: str = "dot",
    ax: Any = None,
) -> tuple[nx.DiGraph, Any]:

    graph = nx.DiGraph()
    node_labels = {}
    node_colors = []

    policy_cmap = plt.cm.Purples  # colour scheme for agent policies
    obs_cmap = plt.cm.Oranges  # colour scheme for agent observations

    if root_node is None:
        root_node = tree.root()

    queue = [(root_node, None, 0)]  # (node, parent_id, depth)
    node_id = 0
    nodes_processed = 0

    while queue:
        current, parent, depth = queue.pop(0)
        nodes_processed += 1
        if depth > max_depth:
            continue

        label_parts = [str(current["idx"])]

        color_map = plt.cm.Greys
        color = to_rgba("lightgrey")

        # observation nodes
        if "observation" in current and current["idx"] != tree.root()["idx"]:
            label_parts.append(f"{observation_description(current['observation'], model)}")

            if "prob" in current:
                label_parts.append(f"P:{formatting_jax(current['prob'])}")

            # add recursive policy score for observation nodes
            if "neg_efe_recursive" in current:
                label_parts.append(
                    f"neg_efe:{formatting_jax(current['neg_efe_recursive'][0])}"
                )
            else:
                label_parts.append(f"neg_efe:{formatting_jax(current['neg_efe'])}")

            color_map = obs_cmap

        # focal agent policy nodes
        elif "policy" in current:
            label_parts.append(f"{action_description(current['policy'], model)}")

            if "prob" in current:
                label_parts.append(f"P:{formatting_jax(current['prob'])}")

            if "neg_efe" in current:
                label_parts.append(f"neg_efe:{formatting_jax(current['neg_efe'])}")

            color_map = policy_cmap

        if "agent" in current:
            # if this is an observation node for a specific agent, use a different color
            # Generate color intensity dynamically for any number of agents
            agent_id = int(current["agent"])
            # Use a range from 0.2 to 0.9 to ensure good contrast
            color_intensity = 0.2 + (0.7 * agent_id / max(1, agent_id + 1))
            color = color_map(color_intensity)
        else:
            color = color_map(0.5)

        node_colors.append(color)

        graph.add_node(
            node_id,
            idx=current["idx"],
            type="policy" if "policy" in current else "observation",
        )
        node_labels[node_id] = "\n".join(label_parts)

        if parent is not None:
            graph.add_edge(parent, node_id)

        # Only process children if we haven't reached max_depth yet
        # This prevents adding children beyond max_depth to the queue
        if depth < max_depth:
            # check for appropriate minimum probability based on node type
            for i, child_idx in enumerate(current.get("children", [])):
                child = tree[child_idx]
                prob = current["children_probs"][i]
                skip_child = False

                if prob < min_prob:
                    skip_child = True
                else:
                    child["prob"] = prob  # add probability to child node

                if not skip_child:
                    queue.append((child, node_id, depth + 1))

        node_id += 1

        if nodes_processed > 1000:
            print("Warning: visualisation is limited to first 1000 nodes")
            break

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    try:
        if layout == "dot":  # traditional hierarchical
            pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
        elif layout == "twopi":  # radial
            pos = nx.nx_agraph.graphviz_layout(graph, prog="twopi")
        else:
            pos = nx.spring_layout(graph, k=0.3, iterations=50)  # default fallback
    except Exception as e:
        print(f"Layout error: {e}. Falling back to spring layout.")
        pos = nx.spring_layout(graph, k=0.3, iterations=50)

    nx.draw_networkx_edges(
        graph, pos, alpha=0.3, width=1, arrows=True, arrowsize=5, ax=ax
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_size,
        node_color=node_colors,
        alpha=0.8,
        linewidths=1,
        edgecolors="gray",
        ax=ax,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        labels=node_labels,
        font_size=font_size,
        verticalalignment="center",
        horizontalalignment="center",
        ax=ax,
    )

    ax.set_axis_off()

    legend_elements = [
        Patch(
            facecolor=policy_cmap(0.5),
            edgecolor="gray",
            alpha=0.8,
            label="Policy Nodes",
        ),
        Patch(
            facecolor=obs_cmap(0.5),
            edgecolor="gray",
            alpha=0.8,
            label="Observation Nodes",
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=font_size)
    return graph, pos


def visualize_plan_tree(
    info: dict[str, Any],
    time_idx: int | None = 0,
    agent_idx: int = 0,
    root_idx: int | None = None,
    model: Any = None,
    observation_description: Callable = observation_to_string,
    action_description: Callable = action_to_string,
    max_depth: int = 4,
    min_prob: float = 0.2,
    layout: str = "dot",
    node_size: int = 1500,
    font_size: int = 10,
    figsize: tuple[int, int] = (8, 7),
    ax: Any = None,
) -> None:
    """
    Wrapper function for plotting plan trees.
    
    Extracts tree from info structure and calls plot_plan_tree with the specified parameters.
    
    Parameters
    ----------
    info : dict
        Information dictionary containing tree data
    time_idx : int or None, optional
        Time index to extract from tree data, by default 0. 
        If None, only extracts by agent_idx as tree is not batched. If specified, extracts by both agent_idx and time_idx.
    agent_idx : int, optional
        Agent index to extract from tree data, by default 0
    root_idx : int, optional
        Index of root node to start plotting from, by default None (uses tree root)
    model : object, optional
        Model object for action/observation descriptions, by default None
    observation_description : callable, optional
        Function to convert observations to strings, by default observation_to_string
    action_description : callable, optional
        Function to convert actions to strings, by default action_to_string
    max_depth : int, optional
        Maximum depth to plot, by default 4
    min_prob : float, optional
        Minimum probability threshold for nodes, by default 0.2
    layout : str, optional
        Layout algorithm ("dot", "twopi"), by default "dot"
    node_size : int, optional
        Size of nodes in the plot, by default 500
    font_size : int, optional
        Font size for node labels, by default 8
    figsize : tuple, optional
        Figure size (width, height), by default (15, 15)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None

    """
    # Extract tree from info structure
    if time_idx is None:
        tree = jtu.tree_map(lambda x: x[agent_idx], info["tree"])
    else:
        tree = jtu.tree_map(lambda x: x[agent_idx, time_idx], info["tree"])
    
    # Handle root node selection
    root_node = None if root_idx is None else tree[root_idx]
    
    # Call plot_plan_tree with extracted tree and parameters
    plot_plan_tree(
        tree,
        model=model,
        root_node=root_node,
        max_depth=max_depth,
        min_prob=min_prob,
        observation_description=observation_description,
        action_description=action_description,
        figsize=figsize,
        font_size=font_size,
        node_size=node_size,
        layout=layout,
        ax=ax,
    )
    
    # Show the plot
    plt.tight_layout()
    plt.show()


def visualize_beliefs(info: dict[str, Any], agent_idx: int = 0, model: Any = None) -> None:
    """Plot the results of the agent's beliefs and actions."""
    num_plots = len(info["qs"])
    fig, axes = plt.subplots(num_plots, 1, figsize=(6, 2*num_plots), sharex=True)

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Assuming all qs[i] have the same width along axis=2
    x_len = info["qs"][0].shape[1]
    x_ticks = np.arange(x_len + 1)

    for i, ax in enumerate(axes):
        if model is not None:
            title = list(model.B[i].event.keys())[0]
        else:
            title = f"state factor {i}"
        ax.set_title(title, fontsize=10)

        # Plot object location beliefs as greyscale intensity
        ax.imshow(info["qs"][i][agent_idx, :, :].T, cmap="gray_r", vmin=0.0, vmax=1.0, aspect='auto')

        ax.set_yticks(jnp.arange(info["qs"][i].shape[-1]))
        if model is not None:
            ax.set_yticklabels(model.B[i].event[list(model.B[i].event.keys())[0]])
        ax.set_xticks(x_ticks)

    # Only bottom subplot shows x-axis labels
    axes[-1].set_xlabel("Time step")
    fig.subplots_adjust(hspace=0.6)  # More vertical spacing
    plt.tight_layout()
    plt.show()


def visualize_env(
    info: dict[str, Any],
    model: Any = None,
    observation_description: Callable = observation_to_string,
    action_description: Callable = action_to_string,
    save_as_gif: bool = False,
    gif_filename: str = "rollout.gif",
) -> None:
    try:
        batch_size = info["env"].num_agents
    except (AttributeError, KeyError):
        batch_size = 1
    num_timesteps = info["qs"][0].shape[1]
    
    # adjust figure size based on number of agents - reduce height to eliminate bottom white space
    base_height = 3.5
    height_per_agent = 1.1
    fig_height = base_height + (batch_size - 1) * height_per_agent
    fig, ax = plt.subplots(figsize=(4, fig_height))
    
    def update(time_idx: int) -> None:
        ax.clear()
        ax.axis("off")
        ax.set_aspect("equal")

        # render the env
        env = jtu.tree_map(lambda x: x[:, time_idx], info["env"])
        ax.imshow(env.render())
        
        # prepare title string for both agents
        title_str = f"Timestep {time_idx}\n"

        base_colours = list(mcolors.TABLEAU_COLORS.keys())
        
        # get the observations and actions for the current timestep
        for agent_idx in range(batch_size): 
            observation = jtu.tree_map(lambda x: x[agent_idx, time_idx][None, ...], info["observation"])
            observation = jnp.concatenate(observation, axis=-1)
            action = jtu.tree_map(lambda x: x[agent_idx, time_idx], info["action"])
            
            obs = observation_description(observation, model).replace("\n", " ")
            act = action_description(action, model).replace("\n", " ")

            # Cycle through available colours, generating more if needed
            if agent_idx < len(base_colours):
                agent_colour = base_colours[agent_idx].replace('tab:', '')
            else:
                # For >10 agents, use HSV to generate additional distinct colours
                import matplotlib.cm as cm
                colour_val = cm.Set3(agent_idx % 12)  # Set3 has 12 colours
                agent_colour = f"RGB{tuple(int(255 * c) for c in colour_val[:3])}"
            
            title_str += f"\n({agent_colour}) Agent {agent_idx} observed ({obs}) \n and selects action ({act}) \n"

        title_y = 1.2 + (batch_size - 1) * 0.125
        ax.text(0.5, title_y, title_str, fontsize=8, 
                ha='center', va='top', transform=ax.transAxes)

    # display the animation
    anim = FuncAnimation(fig, update, frames=num_timesteps, repeat=True, interval=1000)
    
    # the gif gets placed in the bottom of the figure, with the top reserved for title and text
    plt.subplots_adjust(bottom=0.05, top=0.8)

    plt.close(fig)
    display(HTML(anim.to_jshtml()))

    if save_as_gif:
        anim.save(gif_filename, writer="imagemagick", fps=1)
