import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.lax as lax
import jax.random as jr
import jax.nn as nn
import equinox as eqx

from functools import partial
from typing import List

import pymdp
import pymdp.maths
from pymdp.control import (
    compute_info_gain,
    compute_expected_utility,
    compute_expected_state,
    compute_expected_obs,
)


def si_policy_search(
    horizon=5,
    max_nodes=5000,
    max_branching=10,
    policy_prune_threshold=1 / 16,
    observation_prune_threshold=1 / 16,
    entropy_stop_threshold=0.5,
    efe_stop_threshold=1e10,
    kl_threshold=1e-3,
    prune_penalty=512,
    gamma=1,
    topk_obsspace=10000,
):
    """
    Create a search function that can be used in the pymdp `rollout` function.
    """

    @partial(jax.jit, static_argnames=["reset"])
    def search_fn(agent, qs=None, rng_key=None, tree=None, reset=False):
        """
        Infer the best policy given the agent and initial state, optionally provide
        the initial planning tree to continue searching from.

        Returns q_pi and plan tree.
        """
        # WARNING: if we scan over batched agents, then it can happen that
        # the agent has batch_size field set > 1 but the tensors are 
        # in fact of shape [1, ...] because of scanning over it.
        batch_size = agent.A[0].shape[0]

        if tree is None:
            tree = Tree(
                qs, agent.num_factors, agent.num_modalities, max_nodes, max_branching, batch_size=batch_size
            )

        partial_tree_search = partial(
            optimized_tree_search,
            horizon=horizon,
            policy_prune_threshold=policy_prune_threshold,
            observation_prune_threshold=observation_prune_threshold,
            entropy_stop_threshold=entropy_stop_threshold,
            efe_stop_threshold=efe_stop_threshold,
            kl_threshold=kl_threshold,
            prune_penalty=prune_penalty,
            gamma=gamma,
            topk_obsspace=topk_obsspace,
        )

        # vmapping a jax.lax.cond() turns it into a jax.lax.select(),
        # which runs all paths and selects the outcome of the condition
        # this makes it terribly slow for planning trees
        # so we use a scan to iterate over the tree nodes instead
        # tree = jax.vmap(partial_tree_search)(
        #     agent,
        #     tree,
        # )
        policies = agent.policies
        # filter out non-batchsized field (policies)
        agent_filtered, _ = eqx.partition(agent, filter_spec=lambda leaf: leaf.shape[0] == batch_size)
        
        def scan_tree_search(carry, data):
            agent, tree = data
            tree = partial_tree_search(agent, policies, tree)
            return None, (agent, tree)

        if not reset: # reset flag just to create an initial tree structure
            _, data = lax.scan(scan_tree_search, None, (agent_filtered, tree))
            _, tree = data

        # calculate q_pi from the root in a way that is jittable
        q_pi = tree.children_probs[jnp.arange(batch_size), jax.vmap(root_idx)(tree)]
        q_pi = q_pi[:, : agent.policies.shape[-3]]
        return q_pi, { "tree": tree }

    return search_fn


class Tree(eqx.Module):
    """
    A tree structure to hold the planning nodes and their data.

    This is an equinox module which allows for JAX transformations like jit.
    Pre-allocates memory for up to `max_nodes` nodes and `max_branching` children per node.
    """

    size: int = eqx.field(static=True)

    # Tree nodes data (n_nodes, feature_dims)
    # Node belief states
    qs: List[jnp.ndarray]
    # Policy taken if a policy node
    policy: jnp.ndarray
    # Observation expected if an observation node
    observation: jnp.ndarray
    # Node G estimates
    # - for a policy node: G of that policy at that timestep
    # - for an observation node: recursively aggregated G of all children
    G: jnp.ndarray

    # Tree structure bookkeeping
    # Wheter a node is used or not (n_nodes, 1)
    used: jnp.ndarray
    # Which horizon level the node represents (n_nodes, 1)
    horizon: jnp.ndarray
    # Indices to children (n_nodes, max_branching), -1 if unused
    children_indices: jnp.ndarray
    # Probabilities for each child (q_pi for policy children, q_o for observation children)
    # (n_nodes, max_branching)
    children_probs: jnp.ndarray

    def __init__(
        self,
        qs,
        num_action_modalities,
        num_observation_modalities,
        max_nodes,
        max_branching,
        batch_size=1,
        prune_penalty=512
    ):
        self.size = max_nodes

        self.qs = [jnp.zeros((batch_size, max_nodes, 1, q.shape[-1])) for q in qs]
        self.policy = -jnp.ones((batch_size, max_nodes, num_action_modalities), dtype=jnp.int32)
        self.observation = -jnp.ones(
            (batch_size, max_nodes, num_observation_modalities), dtype=jnp.int32
        )
        self.G = jnp.zeros((batch_size, max_nodes, 1))

        self.G = self.G.at[:, -1, :].set(-prune_penalty)

        self.used = jnp.zeros((batch_size, max_nodes, 1), dtype=jnp.bool)
        self.horizon = jnp.zeros((batch_size, max_nodes, 1), dtype=jnp.int32)
        self.children_indices = -jnp.ones((batch_size, max_nodes, max_branching), dtype=jnp.int32)
        self.children_probs = jnp.zeros((batch_size, max_nodes, max_branching), dtype=jnp.float32)

        # set root node
        # self.qs is of shape [batch_size, max_nodes, 1, state_dim]  qs will be [batch_size, 1, state_dim]
        self.qs = jtu.tree_map(lambda x, y: x.at[:, 0, ...].set(y), self.qs, qs)
        self.used = self.used.at[:, 0].set(True)
        self.observation = self.observation.at[:, 0].set(0)

    def __getitem__(self, index):
        """
        Get the node at the given index as a dictionary.

        Works on a non-batched Tree!
        """
        node = {
            "idx": index,
            "qs": jtu.tree_map(lambda x: x[index : index + 1], self.qs),
            "G": self.G[index, 0],
            "horizon": self.horizon[index, 0],
        }

        if jnp.any(self.children_indices[index] >= 0):
            children_indices_array = self.children_indices[index]
            children_probs_array = self.children_probs[index]

            valid_mask = children_indices_array >= 0

            valid_children = children_indices_array[valid_mask]
            valid_probs = children_probs_array[valid_mask]

            node["children"] = valid_children
            node["children_probs"] = valid_probs
        else:
            node["children"] = jnp.array([], dtype=jnp.int32)
            node["children_probs"] = jnp.array([], dtype=jnp.float32)

        policy_valid = jnp.any(self.policy[index] >= 0)
        if policy_valid:
            node["policy"] = self.policy[index]
        else:
            node["observation"] = self.observation[index : index + 1]

        return node

    def root(self):
        """
        Get the root nodes of the tree.

        Works on a non-batched Tree!
        """
        root_node = self[root_idx(self)]
        return root_node


def root_idx(tree):
    root_idx = jnp.argwhere(
        (tree.used)[:, 0]
        & (tree.horizon == 0)[:, 0]
        & ~jnp.all(tree.observation == -1, axis=-1),
        size=1,
        fill_value=-1,
    )[0, 0]
    return root_idx


def step(agent, qs, policies):
    """
    For a given agent, calculate next qs, qo and G given the current qs and policies.
    """

    def _step(q, policy):
        qs = compute_expected_state(q, agent.B, policy, agent.B_dependencies)
        qo = compute_expected_obs(qs, agent.A, agent.A_dependencies)
        u = compute_expected_utility(qo, agent.C)
        ig = compute_info_gain(qs, qo, agent.A, agent.A_dependencies)
        return qs, qo, u, ig

    qs, qo, u, ig = jax.vmap(
        lambda policy: jax.vmap(_step)(qs, policy)
    )(policies)
    G = u + ig
    # jax.debug.print("qs: {qs}", qs=qs)
    # jax.debug.print("qo: {qo}", qo=qo)
    # jax.debug.print("u: {u}", u=u)
    # jax.debug.print("ig: {ig}", ig=ig)
    # jax.debug.print("G from step fn: {G}", G=G)
    return qs, qo, G


def _do_nothing(tree, idx):
    return tree


def _update_node(
    tree,
    idx,
    qs=None,
    policy=None,
    observation=None,
    G=None,
    horizon=None,
    children_indices=None,
    children_probs=None,
):
    """
    Update a node in the planning tree with new data at index `idx`.

    When all nodes are already used, it will not update the node and print a warning.
    """

    def _do_update(
        tree, idx, qs, policy, observation, G, horizon, children_indices, children_probs
    ):
        qs = (
            jtu.tree_map(lambda x, y: x.at[idx].set(y), tree.qs, qs)
            if qs is not None
            else tree.qs
        )
        # if we set a policy, we also set the observation to -1
        if policy is not None:
            policy = tree.policy.at[idx].set(policy)
            observation = tree.observation.at[idx].set(-1)
        elif observation is not None:
            # if observation is set, we also set the policy to -1
            observation = tree.observation.at[idx].set(observation)
            policy = tree.policy.at[idx].set(-1)
        else:
            policy = tree.policy
            observation = tree.observation

        G = tree.G.at[idx].set(G) if G is not None else tree.G
        horizon = (
            tree.horizon.at[idx].set(horizon) if horizon is not None else tree.horizon
        )
        if children_indices is None:
            children_indices = tree.children_indices
        else:
            # pad with -1
            children_indices = jnp.pad(
                children_indices,
                (0, tree.children_indices.shape[-1] - children_indices.shape[0]),
                constant_values=-1,
            )
            children_indices = tree.children_indices.at[idx].set(children_indices)

        if children_probs is None:
            children_probs = tree.children_probs
        else:
            # pad with 0
            children_probs = jnp.pad(
                children_probs,
                (0, tree.children_probs.shape[-1] - children_probs.shape[0]),
                constant_values=0,
            )
            children_probs = tree.children_probs.at[idx].set(children_probs)

        used = tree.used.at[idx].set(True)

        tree = eqx.tree_at(
            lambda x: (
                x.qs,
                x.policy,
                x.observation,
                x.G,
                x.used,
                x.horizon,
                x.children_indices,
                x.children_probs,
            ),
            tree,
            (
                qs,
                policy,
                observation,
                G,
                used,
                horizon,
                children_indices,
                children_probs,
            ),
        )
        return tree

    def _no_op(
        tree, idx, qs, policy, observation, G, horizon, children_indices, children_probs
    ):
        jax.debug.print(
            "WARNING: Used up all {x} nodes in the plan tree...", x=tree.size
        )
        return tree

    # we don't want to update the last node in the array, as it is reserved for all-zeros which child_indices -1 will point to
    return lax.cond(
        idx < tree.size - 1,
        _do_update,
        _no_op,
        tree,
        idx,
        qs,
        policy,
        observation,
        G,
        horizon,
        children_indices,
        children_probs,
    )


def _remove_orphans(tree):
    """
    Remove orphan nodes from the planning tree.
    An orphan node is a used node that is not referenced by any other node as a child.
    The root node is never considered an orphan.
    """

    root_index = root_idx(tree)

    def _is_orphan(tree, idx):
        return (
            (idx != root_index)
            & tree.used[idx, 0]
            & (
                (jnp.any(tree.children_indices == idx, axis=-1) & tree.used[:, 0]).sum()
                == 0
            )
        )

    def _has_orphans(tree):
        # search for used nodes that never show up as a child
        valid_mask = tree.children_indices != -1
        safe_children = jnp.where(valid_mask, tree.children_indices, 0) * tree.used
        flat_children = safe_children.flatten()
        flat_valid_mask = valid_mask.flatten()
        counts = (
            jnp.zeros(tree.size, dtype=bool)
            .at[flat_children]
            .add(flat_valid_mask.astype(jnp.int32))
        )
        is_referenced = counts > 0
        orphans = jnp.logical_and(tree.used[:, 0], jnp.logical_not(is_referenced))
        orphans = orphans.at[root_index].set(False)  # root node is never an orphan
        return orphans.any()

    def _remove_orphan(tree, idx):
        def _noop(t, idx):
            return t

        def _remove_orphan(t, idx):
            # jax.debug.print("remove orphan {idx}", idx=(idx, t.used[idx, 0], _is_orphan(t, idx), root_idx))
            t = eqx.tree_at(
                lambda x: x.used,
                t,
                t.used.at[idx].set(False),
            )
            return t

        tree = lax.cond(
            _is_orphan(tree, idx),
            _remove_orphan,
            _noop,
            tree,
            idx,
        )

        return tree, None

    def _scan_for_orphans(tree):
        tree, _ = lax.scan(_remove_orphan, tree, jnp.arange(tree.size))
        return tree

    tree = lax.while_loop(_has_orphans, _scan_for_orphans, tree)
    return tree


@partial(jax.jit, static_argnums=(0))
def _calculate_probabilities(return_size, topk_probs):
    """
    Calculate joint probabilities using stride-based indexing with top-k probabilities.
    
    Args:
        return_size: Number of observation combinations to generate.
        topk_probs: List of top-k probability arrays, one per modality.
    """
    joint_probs = jnp.ones(return_size)
    
    # Use topk_probs directly with per-modality k-based stride indexing
    stride = 1
    strides = []
    k_values = [len(probs) for probs in topk_probs]
    
    # Calculate strides for each dimension (rightmost changes fastest)
    for k in reversed(k_values):
        strides.insert(0, stride)
        stride *= k
        
    # For each modality, select from top-k probabilities
    for i, (stride, k) in enumerate(zip(strides, k_values)):
        indices = jnp.arange(return_size)
        # Convert linear index to top-k coordinate for this modality
        topk_coords = (indices // stride) % k
        selected_probs = topk_probs[i][topk_coords]
        joint_probs = joint_probs * selected_probs
        
    return joint_probs


@partial(jax.jit, static_argnums=(0, 1))
def _generate_observations(shapes, return_size, topk_indices):
    """
    Generate observation combinations using stride-based indexing with top-k indices.
    
    Args:
        shapes: Shape of each modality in the observation space.
        return_size: Number of observation combinations to generate.
        topk_indices: List of top-k index arrays, one per modality.
    """
    combinations = jnp.zeros((return_size, len(shapes)), dtype=jnp.int32)
    
    # Generate in top-k space, then map to original indices
    k_values = [len(indices) for indices in topk_indices]
    stride = 1
    
    for i, k in enumerate(reversed(k_values)):
        pos = len(shapes) - 1 - i
        # Generate combinations in [0, k-1] space
        topk_coords = (jnp.arange(return_size) // stride) % k
        # Map to original observation indices
        original_coords = topk_indices[pos][topk_coords]
        combinations = combinations.at[:, pos].set(original_coords)
        stride *= k
        
    return combinations


def optimized_tree_search(
    agent,
    policies,
    tree,
    horizon,
    policy_prune_threshold=1 / 16,
    observation_prune_threshold=1 / 16,
    entropy_stop_threshold=0.5,
    efe_stop_threshold=1e10,
    kl_threshold=1e-3,
    prune_penalty=512,
    gamma=1,
    topk_obsspace=10000,
):
    """
    Perform a sophisticated inference tree search given an agent and planning tree.

    Keeps expanding the tree until one of the following conditions is met:
    - The horizon is reached.
    - The entropy of the root node's policy distribution is below a threshold.
    - The expected free energy of the root node is below a threshold.

    Args:
        agent: The agent to use for planning.
        tree: The initial planning tree.
        horizon: The maximum horizon to expand the tree.
        policy_prune_threshold: Threshold for pruning policies.
        observation_prune_threshold: Threshold for pruning observations.
        entropy_stop_threshold: Entropy threshold to stop expanding.
        efe_stop_threshold: Expected free energy threshold to stop expanding.
        kl_threshold: KL divergence threshold for reusing nodes.
        prune_penalty: Penalty for pruning a node.
        gamma: Precison of q_pi.

    Returns:
        tree: The expanded planning tree.
    """

    def _expand_observation_nodes(tree, data):
        """
        Expand a policy node into new observation nodes.

        If the policy node has a probability above the `policy_prune_threshold`, we futher expand
        """
        policy_node_idx, qs_next, qo, prob = data

        def add_observation_nodes(tree, policy_node_idx, qs, qo, topk_obsspace):
            """Generate observation combinations with top-k filtering."""
            shapes = [o.shape[-1] for o in qo]
            
            k = topk_obsspace
            
            def get_topk_for_factor(factor_probs):
                # Use effective k to handle cases where k exceeds modality size bc default k is 10000
                modality_size = factor_probs[0].shape[0]
                k_effective = min(k, modality_size)
                # Get top k indices and their probabilities for this modality
                top_probs, top_indices = jax.lax.top_k(factor_probs[0], k_effective)
                # Renormalise the top probabilities # TODO: check that this is ok to do
                top_probs = top_probs / jnp.sum(top_probs)
                return top_indices, top_probs
            
            # Extract top-k data for each modality
            topk_data = [get_topk_for_factor(factor) for factor in qo]
            topk_indices = [data[0] for data in topk_data]
            topk_probs = [data[1] for data in topk_data]
            
            # Calculate actual number of combinations based on effective k values
            k_effective_per_modality = [len(indices) for indices in topk_indices]
            num_combinations = int(np.prod(k_effective_per_modality))
            
            # Generate combinations using refactored helper functions
            observations = _generate_observations(tuple(shapes), num_combinations, topk_indices)
            probabilities = _calculate_probabilities(num_combinations, topk_probs)

            def add_observation_node(tree, data):
                observation, prob = data

                def consider_add(t):
                    # calculate posterior state belief and check if
                    # we need to add a new observation node

                    # convert to correct dimensions for infer_states
                    obs = [o[None, None, ...] for o in observation]
                    agent_expanded = jtu.tree_map(lambda x: x[None, ...], agent)
                    qs_post = agent_expanded.infer_states(obs, qs)
                    # remove time dim from qs_post
                    qs_post = jtu.tree_map(lambda x: x[:, 0, :], qs_post)

                    # check if we already have a node with this belief (or close enough)
                    def diff_kl(qs_orig, qs_post):
                        d = jtu.tree_map(
                            lambda x, y: jnp.sum(
                                x
                                * (
                                    jnp.log(jnp.clip(x, 1e-10, 1))
                                    - jnp.log(jnp.clip(y[0], 1e-10, 1))
                                ),
                                axis=-1,
                            ),
                            qs_orig,
                            qs_post,
                        )
                        d = jtu.tree_reduce(lambda x, y: x + y, d, 0)[:, 0]

                        mask = (t.used)[:, 0]  # only consider used nodes
                        mask = mask & jnp.all(
                            tree.policy < 0, axis=-1
                        )  # only consider observation nodes

                        # TODO only consider nodes with smaller horizon?
                        # i.e. you can get to this state earlier
                        # or should we compare e.g. expected free energy of all paths leading
                        # to same posterior?
                        mask = mask & (t.horizon[:, 0] < t.horizon[policy_node_idx, 0])

                        return d * mask + 1e10 * (1 - mask)

                    kl = diff_kl(t.qs, qs_post)

                    def really_add(t, d):
                        new_idx = jnp.where(
                            ~t.used[:, 0], jnp.arange(t.size), t.size
                        ).min()

                        # jax.debug.print("Add new obs node {x}", x=new_idx)

                        t = _update_node(
                            t,
                            new_idx,
                            qs=qs_post,
                            observation=observation,
                            horizon=t.horizon[policy_node_idx, 0],
                            G=0,
                            children_indices=jnp.empty((0,)),
                            children_probs=jnp.empty((0,)),
                        )
                        return t, new_idx

                    def skip_add(t, kl):
                        # new_idx = jnp.where(
                        #     ~t.used[:, 0], jnp.arange(t.size), t.size
                        # ).min()
                        # jax.debug.print("skipped due to kl {kl} at {idx}", kl=kl.min(), idx = new_idx)
                        # if we already have a node with this belief, stop expanding
                        # new_idx = jnp.where(
                        #     ~t.used[:, 0], jnp.arange(t.size), t.size
                        # ).min()
                        # jax.debug.print("Skip adding observation node {x} as it is already in the tree as {y}",x=new_idx, y=kl.argmin())
                        # we return the original tree and -1 to indicate no new node was added
                        return t, -1

                    return lax.cond(
                        kl.min() <= kl_threshold, skip_add, really_add, t, kl
                    )

                def no_op(t):
                    new_idx = jnp.where(
                            ~t.used[:, 0], jnp.arange(t.size), t.size
                        ).min()
                    
                    return t, -1

                tree, obs_idx = lax.cond(
                    prob > observation_prune_threshold,
                    consider_add,
                    no_op,
                    tree,
                )
                return tree, obs_idx

            tree, obs_indices = lax.scan(
                add_observation_node, tree, (observations, probabilities)
            )

            # update policy parent with child indices
            # get the indices to select, pad with -1
            indices = jnp.where(
                obs_indices >= 0,
                size=tree.children_indices.shape[-1],
                fill_value=-1,
            )[0]
            obs_indices = jnp.concatenate([obs_indices, -jnp.ones(1)])
            obs_indices = obs_indices[indices]

            obs_probabilities = jnp.concatenate([probabilities, jnp.zeros(1)])
            obs_probabilities = obs_probabilities[indices]
            tree = _update_node(
                tree,
                policy_node_idx,
                children_indices=obs_indices,
                children_probs=obs_probabilities,
            )
            return tree

        def no_op(tree, policy_node_idx, qs, qo):
            return tree

        # Create partial function that includes topk_obsspace parameter
        add_observation_nodes_partial = lambda tree, policy_node_idx, qs, qo: add_observation_nodes(tree, policy_node_idx, qs, qo, topk_obsspace)

        tree = lax.cond(
            prob[0] > policy_prune_threshold,
            add_observation_nodes_partial,
            no_op,
            tree,
            policy_node_idx,
            qs_next,
            qo,
        )
        return tree, None

    def _expand_policy_nodes(t, idx):
        """
        Given an observation node at `idx`, expand into new policy nodes.

        This will calculate the expected states, outcomes and free energy for all policies,
        and then expand the tree with new policy nodes.

        For each policy that is above the `policy_prune_threshold`, it will
        also expand into new observation nodes.
        """

        # jax.debug.print("Expand policies of node {idx} at horizon {h}", idx=idx, h=t.horizon[idx, 0])
        # calculate expected states, outcomes and free energy for all policies
        qs_current = jtu.tree_map(lambda x: x[idx], t.qs)
        qs_next, qo, G = step(agent, qs_current, policies)
        q_pi = nn.softmax(G * gamma, axis=0)

        # expand policy nodes
        def add_policy_node(tree, data):
            policy, qs_next, prob, G = data

            def really_add(tree, policy, qs_next, G):
                new_idx = jnp.where(
                    ~tree.used[:, 0], jnp.arange(tree.size), tree.size
                ).min()

                # jax.debug.print("Add policy node {x}", x=new_idx)
                # jax.debug.print("G: {G}", G=G)

                tree = _update_node(
                    tree,
                    new_idx,
                    qs=qs_next,
                    policy=policy,
                    G=G,
                    horizon=tree.horizon[idx, 0] + 1,
                    children_indices=jnp.empty((0,)),
                    children_probs=jnp.empty((0,)),
                )
                return tree, new_idx

            def skip_add(tree, policy, qs_next, G):
                # jax.debug.print("skip add policy node {p} as the prob is {pr}", p=policy, pr=prob[0])
                return tree, -1

            return lax.cond(
                prob[0] > policy_prune_threshold,
                really_add,
                skip_add,
                tree,
                policy,
                qs_next,
                G,
            )

        # policies is of shape (n_policies, timesteps (=1), n_actions)
        t, policy_indices = lax.scan(add_policy_node, t, (policies[:, 0, :], qs_next, q_pi, G))
        # jax.debug.print("q_pi: {q_pi}", q_pi=q_pi)

        # update parent with child indices
        t = _update_node(
            t, idx, children_indices=policy_indices, children_probs=q_pi[:, 0]
        )

        # now expand observation nodes
        t, _ = lax.scan(
            _expand_observation_nodes, t, (policy_indices, qs_next, qo, q_pi)
        )

        return t

    def _expand_node(carry, idx):
        """
        Consider expansion for a node at `idx`. We call this function for every node in the tree,
        allowing us to efficiently jit and scan.

        Every pass, we will consider (used) observation nodes that are leaves from the tree (i.e. at
        a given horizon) and expand them into new policy (and observation) nodes.
        """
        tree, agent, h = carry

        tree = lax.cond(
            (tree.used.sum() < tree.used.shape[0] - 1)
            & (tree.used[idx, 0])
            & (tree.horizon[idx, 0] == h)
            & (tree.observation[idx, 0] >= 0)
            & (tree.used.sum() < tree.size),
            _expand_policy_nodes,
            _do_nothing,
            tree,
            idx,
        )

        return (tree, agent, h), None

    def _backward_node(tree, idx):
        """
        Run a backward pass on an observation node at `idx`.

        Will iterate over all policy children of this observation node,
        gather their G values and probabilities, and recursively
        aggregate them to calculate the G value for this observation node.
        """
        # jax.debug.print("Backward node {idx}", idx=idx)

        def recursive_G(t, policy_idx):
            observation_nodes = t.children_indices[policy_idx]
            probabilities = t.children_probs[policy_idx]

            def sum_over_obs(carry, obs_idx, p):
                carry += t.G[obs_idx, 0] * p
                return carry

            def zero(carry, obs_idx, p):
                return carry

            def accumulate_g(carry, data):
                obs_idx, prob = data
                return (
                    lax.cond(obs_idx >= 0, sum_over_obs, zero, carry, obs_idx, prob),
                    None,
                )

            # accumulate G for all observation nodes
            G_acc, _ = lax.scan(accumulate_g, 0.0, (observation_nodes, probabilities))

            # and also add G of this policy timestep
            G_acc += t.G[policy_idx, 0]

            return t, G_acc

        def add_prune_penalty(t, policy_idx):
            return t, jnp.asarray(-prune_penalty, dtype=jnp.float32)

        def backward(t, idx):
            return lax.cond(
                # only accumulate G if it is a policy node with children, use prune penalty otherwise
                (idx >= 0) & (jnp.any(t.children_indices[idx] >= 0)),
                recursive_G,
                add_prune_penalty,
                t,
                idx,
            )

        # iterate over all policy children
        tree, G_recursive = lax.scan(backward, tree, tree.children_indices[idx])

        # calculate new q_pi and recursive G
        q_pi = nn.softmax(G_recursive * gamma, axis=0)
        G_recursive = jnp.dot(q_pi, G_recursive)

        # update G for this observation node
        # jax.debug.print(
        #     "Update node {idx} with G {G_recursive}", idx=idx, G_recursive=G_recursive
        # )

        # jax.debug.print("G in backward: {G_recursive}", G_recursive=G_recursive)
        # jax.debug.print("q_pi in backward: {q_pi}", q_pi=q_pi)

        # prune children if their q_pi is below the threshold
        prune_mask = q_pi < policy_prune_threshold
        children_indices = (1-prune_mask) * tree.children_indices[idx] + prune_mask * -1

        tree = _update_node(tree, idx, G=G_recursive, children_indices=children_indices, children_probs=q_pi)
        return tree

    def _tree_backward(tree, h):
        """
        Calculate the new recursive G value for all nodes at horizon h
        """

        # jax.debug.print("Go backward at horizon {h}", h=h)

        def _do_backward(tree, idx):
            tree = lax.cond(
                (tree.used[idx, 0])
                & (tree.horizon[idx, 0] == h)
                & (tree.observation[idx, 0] >= 0),
                _backward_node,
                _do_nothing,
                tree,
                idx,
            )
            return tree, None

        # update G for all nodes at this horizon
        tree, _ = lax.scan(
            _do_backward, tree, jnp.arange(tree.used.shape[0] - 1, -1, -1)
        )

        # remove orphans if any by pruning
        tree = _remove_orphans(tree)
        return tree, h - 1

    def _expand_horizon(tree, agent, h):
        """
        Expand the planning tree at a given horizon by expanding all nodes at that horizon.
        After expanding, it will also perform a backward pass to recursively update the G values
        """

        # jax.debug.print("Expanding horizon {h}", h=h)

        # expand all nodes at this horizon
        (tree_updated, agent, _), _ = lax.scan(
            _expand_node, (tree, agent, h), jnp.arange(tree.used.shape[0])
        )

        # jax.debug.print("Got {n} nodes at horizon {h}", n=tree_updated.used.sum(), h=h)

        # TODO backward here or backward at the end?
        # tree backward
        def backward_pass(tree, agent, h):
            def backward_step(carry, current_h):
                def do_backward(t, ch):
                    return _tree_backward(t, ch)
                
                def skip_backward(t, ch):
                    return t, ch - 1
                
                return lax.cond(
                    current_h <= h,
                    do_backward,
                    skip_backward,
                    carry,
                    current_h
                )

            # scan from max horizon down to 0 (inclusive)
            tree_bw, _ = lax.scan(backward_step, tree, jnp.arange(horizon, -1, -1))
            
            # jax.debug.print("Got {n} nodes after tree backward", n=tree_bw.used.sum())
            return tree_bw, agent, h + 1

        # when we run out of nodes, stop expanding and return tree at prev horizon
        return lax.cond(tree_updated.used.sum() < tree.used.shape[0] - 1, backward_pass, lambda x,y,z: (tree, agent, horizon), tree_updated, agent, h)

    # expand till max horizon (h) 
    def continue_expansion(tree, agent, h):
        q_pi = tree.children_probs[root_idx(tree)]
        entropy = pymdp.maths.stable_entropy(q_pi)

        # jax.debug.print("horizon {h} out of {max_h}, entropy={e:.3f}, entropy threshold = {et}, halted_entropy={estop}, efe={g:.3f}, efe threshold = {efet}, halted_efe={efestop}, jnp.all(q_pi == 0) = {qpi}",
        #                 h=h, max_h=horizon, e=entropy, et=entropy_stop_threshold,
        #                 estop=entropy <= entropy_stop_threshold, g=tree.G[tree.root_idx(), 0], efet=efe_stop_threshold,
        #                 efestop=tree.G[tree.root_idx(), 0] >= efe_stop_threshold, qpi=jnp.all(q_pi == 0))

        return (
            (h < horizon)
            & (jnp.all(q_pi == 0) | (entropy > entropy_stop_threshold))
            & (tree.G[root_idx(tree), 0] < efe_stop_threshold)
        )

    def scan_step(carry, _):
        tree, agent, h, stop_expansion = carry
        
        def do_expand(args):
            tree, agent, h = args
            return _expand_horizon(tree, agent, h)
        
        def do_nothing(args):
            tree, agent, h = args
            return tree, agent, h
        
        # expand if we haven't stopped based on stop criteria (such as entropy or EFE threshold) yet
        tree, agent, h = lax.cond(
            stop_expansion,
            do_nothing,
            do_expand,
            (tree, agent, h)
        )
        
        # check if we should stop next iteration (stay stopped if stop criteria is met)
        stop_expansion = stop_expansion | ~continue_expansion(tree, agent, h)
        
        return (tree, agent, h, stop_expansion), None

    # run scan for max horizon iterations
    initial_h = (tree.horizon * tree.used).max()
    initial_carry = (tree, agent, initial_h, False)
    (tree, agent, _, _), _ = lax.scan(scan_step, initial_carry, None, length=horizon)

    return tree
