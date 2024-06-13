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
        return [node for node in self.nodes if "qs" in node.keys() and len(node["children"]) == 0]

    def append(self, node):
        self.nodes.append(node)


def step(agent, qs, policies, gamma=32):
    def _step(a, b, c, q, policy):
        qs = compute_expected_state(q, b, policy, agent.B_dependencies)
        qo = compute_expected_obs(qs, a, agent.A_dependencies)
        u = compute_expected_utility(qo, c)
        ig = compute_info_gain(qs, qo, a, agent.A_dependencies)
        return qs, qo, u, ig

    qs, qo, u, ig = vmap(lambda policy: vmap(_step)(agent.A, agent.B, agent.C, qs, policy))(policies)
    G = u + ig
    return qs, qo, nn.softmax(G * gamma, axis=0), G


def tree_search(
    agent,
    qs,
    horizon,
    policy_prune_threshold=1 / 16,
    policy_prune_topk=-1,
    observation_prune_threshold=1 / 16,
    entropy_prune_threshold=0.5,
    prune_penalty=512,
):
    root_node = {
        "qs": jtu.tree_map(lambda x: x[:, -1, ...], qs),
        "G_t": 0.0,
        "parent": None,
        "children": [],
        "n": 0,
    }
    tree = Tree(root_node)

    for _ in range(horizon):

        leaves = tree.leaves()
        qs_leaves = stack_leaves([leaf["qs"] for leaf in leaves])
        qs_pi, qo_pi, q_pi, G = vmap(lambda leaf: step(agent, leaf, agent.policies))(qs_leaves)

        for l, node in enumerate(leaves):
            tree = expand_node(
                agent,
                node,
                tree,
                jtu.tree_map(lambda x: x[l, ...], qs_pi),
                jtu.tree_map(lambda x: x[l, ...], qo_pi),
                q_pi[l],
                G[l],
                policy_prune_threshold,
                policy_prune_topk,
                observation_prune_threshold,
                prune_penalty,
            )

        if policy_entropy(tree.root()) < entropy_prune_threshold:
            break

    return tree


def expand_node(
    agent,
    node,
    tree,
    qs_pi,
    qo_pi,
    q_pi,
    G,
    policy_prune_threshold=1 / 16,
    policy_prune_topk=-1,
    observation_prune_threshold=1 / 16,
    prune_penalty=512,
    gamma=32,
):
    policies = agent.policies

    node["policies"] = policies
    node["q_pi"] = q_pi[:, 0]
    node["G"] = jnp.array([jnp.dot(q_pi[:, 0], G[:, 0])])
    node["children"] = []

    ordered = jnp.argsort(q_pi[:, 0])[::-1]
    policies_to_consider = []
    for idx in ordered:
        if policy_prune_topk > 0 and len(policies_to_consider) >= policy_prune_topk:
            break
        if q_pi[idx] >= policy_prune_threshold:
            policies_to_consider.append(idx)
        else:
            break

    observations, qs_priors, probs, policy_nodes = [], [], [], []
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

        if idx in policies_to_consider:
            tree.append(policy_node)

        if idx in policies_to_consider:
            # branch over possible observations
            qo_next = jtu.tree_map(lambda x: x[idx][0], qo_pi)
            qs_prior = jtu.tree_map(lambda x: x[idx], qs_pi)

            # TODO: wip
            # shapes = [s.shape[0] for s in qo_next]
            # combinations = jnp.array(list(itertools.product(*[jnp.arange(s) for s in shapes])))

            # def calculate_prob(combination):
            #     return jnp.prod(jnp.array([qo_next[i][combination[i]] for i in range(len(combination))]))

            # prob = jax.vmap(calculate_prob)(combinations)

            # valid_indices = prob >= observation_prune_threshold
            # valid_combinations = combinations[valid_indices]
            # observation = [jnp.array([[k[i]] for k in valid_combinations]) for i in range(len(qo_next))]

            # observations.append(observation)
            # qs_priors.append(qs_prior)
            # probs.append(prob[valid_indices])
            # policy_nodes.append(policy_node)

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

                observations.append(observation)
                qs_priors.append(qs_prior)
                probs.append(prob)
                policy_nodes.append(policy_node)

    stacked_observations = stack_leaves(observations)
    stacked_qs_priors = stack_leaves(qs_priors)
    qs_next = vmap(agent.infer_states)(stacked_observations, None, stacked_qs_priors, None)

    for idx, observation in enumerate(observations):
        observation_node = {
            "observation": observation,
            "prob": probs[idx],
            "qs": jtu.tree_map(lambda x: x[idx, :, 0, ...], qs_next),
            "G": jnp.zeros((1)),
            "parent": policy_nodes[idx],
            "children": [],
            "n": node["n"] + 1,
        }
        policy_nodes[idx]["children"].append(observation_node)
        tree.append(observation_node)

    tree_backward(node, prune_penalty, gamma)
    return tree


def tree_backward(node, prune_penalty=512, gamma=32):
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


def policy_entropy(node):
    return -jnp.dot(node["q_pi"], jnp.log(node["q_pi"] + pymdp.maths.EPS_VAL))


def stack_leaves(data):
    return [jnp.stack([d[i] for d in data]) for i in range(len(data[0]))]


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
