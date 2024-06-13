import itertools
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn
from jax import vmap

import pymdp
from pymdp.jax.control import compute_info_gain, compute_expected_utility, compute_expected_state, compute_expected_obs


class Tree:
    def __init__(self, root):
        self.nodes = [root]

    def root(self):
        return self.nodes[0]

    def children(self, node):
        return node["children"]

    def parent(self, node):
        return node["parent"]

    def leaves(self):
        return [node for node in self.nodes if len(node["children"]) == 0]

    def append(self, node):
        self.nodes.append(node)


# def step(agent, qs, policies, gamma=32):

#     def _step(a, b, c, q, policy):
#         qs = compute_expected_state(q, b, policy, agent.B_dependencies)
#         qo = compute_expected_obs(qs, a, agent.A_dependencies)
#         u = compute_expected_utility(qo, c)
#         ig = compute_info_gain(qs, qo, a, agent.A_dependencies)
#         return qs, qo, u, ig

#     qs, qo, u, ig = vmap(lambda policy: vmap(_step)(agent.A, agent.B, agent.C, qs, policy))(policies)
#     G = u + ig
#     return qs, qo, nn.softmax(G * gamma, axis=0), G


# def tree_search(agent, qs, horizon):
#     root_node = {
#         "qs": jtu.tree_map(lambda x: x[:, -1, ...], qs),
#         "parent": None,
#         "children": [],
#         "n": 0,
#     }
#     tree = Tree(root_node)

#     # TODO: scan here
#     for _ in range(horizon):
#         leaves = tree.leaves()
#         qs_leaves = stack_data([leaf["qs"] for leaf in leaves])
#         qs_pi, _, q_pi, G = vmap(lambda leaf: step(agent, leaf, agent.policies))(qs_leaves)
#         tree = expand_tree(tree, leaves, agent.policies, qs_pi, q_pi, G)
#     return tree


# def expand_tree(tree, leaves, policies, qs_pi, q_pi, G):
#     # TODO: we will vmap this at some point
#     for l, leaf in enumerate(leaves):
#         for p, policy in enumerate(policies):
#             child = {
#                 "policy": policy,
#                 "q_pi": q_pi[l, p],
#                 "qs": jtu.tree_map(lambda x: x[l, p, ...], qs_pi),
#                 "G": G[l, p],
#                 "parent": leaf,
#                 "children": [],
#                 "n": leaf["n"] + 1,
#             }
#             leaf["children"].append(child)
#             tree.append(child)
#     return tree


# def stack_data(data):
#     return [jnp.stack([d[i] for d in data]) for i in range(len(data[0]))]


# def tree_search(agent, qs, planning_horizon):
#     # cut out time dimension
#     qs = jtu.tree_map(lambda x: x[:, -1, ...], qs)
#     root_node = {
#         "qs": qs,
#         "parent": None,
#         "children": [],
#         "n": 0,
#     }
#     tree = Tree(root_node)

#     h = 0
#     while h < planning_horizon:
#         # TODO refactor so we can vectorize this
#         for node in tree.leaves():
#             tree = expand_node_vanilla(agent, node, tree)

#         h += 1

#     return tree


def step(agent, qs, policies):

    def _step(a, b, c, qs, policy):
        qs_pi = compute_expected_state(qs, b, policy, agent.B_dependencies)
        qo_pi = compute_expected_obs(qs_pi, a, agent.A_dependencies)
        u = compute_expected_utility(qo_pi, c)
        ig = compute_info_gain(qs_pi, qo_pi, a, agent.A_dependencies)
        return qs_pi, qo_pi, u, ig

    qs_pi, qo_pi, u, ig = vmap(lambda policy: vmap(_step)(agent.A, agent.B, agent.C, qs, policy))(policies)
    G = u + ig
    return qs_pi, qo_pi, G, u, ig


def tree_search(
    agent,
    qs,
    planning_horizon,
    policy_prune_threshold=1 / 16,
    policy_prune_topk=-1,
    observation_prune_threshold=1 / 16,
    entropy_prune_threshold=0.5,
    prune_penalty=512,
):
    # cut out time dimension
    qs = jtu.tree_map(lambda x: x[:, -1, ...], qs)
    root_node = {
        "qs": qs,
        "G_t": 0.0,
        "parent": None,
        "children": [],
        "n": 0,
    }
    tree = Tree(root_node)

    h = 0
    while h < planning_horizon:
        # TODO refactor so we can vectorize this

        # TODO add some early stopping based on q_pi entropy
        if "q_pi" in tree.root().keys():
            q_pi = tree.root()["q_pi"]
            H = -jnp.dot(q_pi, jnp.log(q_pi + pymdp.maths.EPS_VAL))
            if H < entropy_prune_threshold:
                break

        for node in tree.leaves():
            tree = expand_node_sophisticated(
                agent, node, tree, policy_prune_threshold, policy_prune_topk, observation_prune_threshold, prune_penalty
            )

        h += 1

    return tree


def expand_node_vanilla(agent, node, tree, gamma=32):
    qs = node["qs"]
    policies = agent.policies

    qs_pi, qo_pi, G, u, ig = step(agent, qs, policies)
    q_pi = nn.softmax(G * gamma, axis=0)

    node["policies"] = policies
    node["q_pi"] = q_pi[:, 0]
    node["G"] = jnp.array([jnp.dot(q_pi[:, 0], G[:, 0])])

    for idx in range(policies.shape[0]):
        policy_node = {
            "policy": policies[idx, 0],
            "prob": q_pi[idx, 0],
            "qs": jtu.tree_map(lambda x: x[idx, ...], qs_pi),
            "qo": jtu.tree_map(lambda x: x[idx, ...], qo_pi),
            "G_t": G[idx],
            "G": G[idx],
            "parent": node,
            "children": [],
            "n": node["n"] + 1,
        }

        node["children"].append(policy_node)
        tree.append(policy_node)

    # update G of parents
    while node["parent"] is not None:
        parent = node["parent"]
        G_children = jnp.array([child["G"][0] for child in parent["children"]])
        q_pi = nn.softmax(G_children * gamma)

        G = jnp.dot(q_pi, G_children) + parent["G_t"]
        parent["G"] = G
        parent["q_pi"] = q_pi

        for idx, c in enumerate(parent["children"]):
            c["prob"] = q_pi[idx]
        node = parent

    return tree


def expand_node_sophisticated(
    agent,
    node,
    tree,
    policy_prune_threshold=1 / 16,
    policy_prune_topk=-1,
    observation_prune_threshold=1 / 16,
    prune_penalty=512,
    gamma=32,
):
    qs = node["qs"]
    policies = agent.policies

    qs_pi, qo_pi, G, u, ig = step(agent, qs, policies)
    q_pi = nn.softmax(G * gamma, axis=0)

    node["policies"] = policies
    node["q_pi"] = q_pi[:, 0]
    node["G"] = jnp.array([jnp.dot(q_pi[:, 0], G[:, 0])])
    node["children"] = []

    # expand the policies and observations of this node
    ordered = jnp.argsort(q_pi[:, 0])[::-1]
    policies_to_consider = []
    for idx in ordered:
        if policy_prune_topk > 0 and len(policies_to_consider) >= policy_prune_topk:
            break
        if q_pi[idx] >= policy_prune_threshold:
            policies_to_consider.append(idx)
        else:
            break

    for idx in range(len(policies)):
        policy_node = {
            "policy": policies[idx, 0],
            "prob": q_pi[idx, 0],
            "G_t": G[idx],
            "parent": node,
            "children": [],
            "n": node["n"] + 1,
        }
        node["children"].append(policy_node)
        tree.append(policy_node)

        if idx in policies_to_consider:
            # branch over possible observations
            qo_next = jtu.tree_map(lambda x: x[idx][0], qo_pi)

            for k in itertools.product(*[range(s.shape[0]) for s in qo_next]):
                prob = 1.0
                for i in range(len(k)):
                    prob *= qo_next[i][k[i]]

                # ignore low probability observations in the search tree
                if prob < observation_prune_threshold:
                    continue

                # qo_one_hot = []
                observation = []
                for i in range(len(qo_next)):
                    observation.append(jnp.array([[k[i]]]))

                qs_prior = jtu.tree_map(lambda x: x[idx], qs_pi)

                qs_next = agent.infer_states(
                    observations=observation,
                    past_actions=None,
                    empirical_prior=qs_prior,
                    qs_hist=None,
                )

                observation_node = {
                    "observation": observation,
                    "prob": prob,
                    "qs": jtu.tree_map(lambda x: x[:, 0, ...], qs_next),
                    "G": jnp.zeros((1)),
                    "parent": policy_node,
                    "children": [],
                    "n": node["n"] + 1,
                }
                policy_node["children"].append(observation_node)
                tree.append(observation_node)

    # update G of parents
    while node["parent"] is not None:
        parent = node["parent"]["parent"]
        G_children = jnp.zeros(len(node["children"]))

        for idx, n in enumerate(parent["children"]):
            # iterate over policy nodes
            G_children = G_children.at[idx].add(n["G_t"][0])

            if len(n["children"]) == 0:
                # add penalty to pruned nodes
                G_children = G_children.at[idx].add(-prune_penalty)
            else:
                # sum over all likely observations
                for o in n["children"]:
                    prob = o["prob"]
                    G_children = G_children.at[idx].add(o["G"][0] * prob)

        # update parent node
        q_pi = nn.softmax(G_children * gamma)
        G = jnp.array([jnp.dot(q_pi, G_children)])
        parent["G"] = G
        parent["q_pi"] = q_pi

        for idx, c in enumerate(parent["children"]):
            c["prob"] = q_pi[idx]

        node = parent

    return tree
