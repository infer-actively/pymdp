import itertools
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import nn
from jax import vmap

from pymdp.jax.control import compute_info_gain, compute_expected_utility, compute_expected_state, compute_expected_obs


class Tree:
    # TODO - placeholder tree class, replace with jaxified version later

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


def tree_search(agent, qs, horizon):
    root_node = {
        "qs": jtu.tree_map(lambda x: x[:, -1, ...], qs),
        "parent": None,
        "children": [],
        "n": 0,
    }
    tree = Tree(root_node)

    # TODO: scan here
    for _ in range(horizon):
        leaves = tree.leaves()
        qs_leaves = stack_data([leaf["qs"] for leaf in leaves])
        qs_pi, _, q_pi, G = vmap(lambda leaf: step(agent, leaf, agent.policies))(qs_leaves)
        tree = expand_tree(tree, leaves, agent.policies, qs_pi, q_pi, G)
    return tree


def expand_tree(tree, leaves, policies, qs_pi, q_pi, G):
    # TODO: we will vmap this at some point
    for l, leaf in enumerate(leaves):
        for p, policy in enumerate(policies):
            child = {
                "policy": policy,
                "q_pi": q_pi[l, p],
                "qs": jtu.tree_map(lambda x: x[l, p, ...], qs_pi),
                "G": G[l, p],
                "parent": leaf,
                "children": [],
                "n": leaf["n"] + 1,
            }
            leaf["children"].append(child)
            tree.append(child)
    return tree


def stack_data(data):
    return [jnp.stack([d[i] for d in data]) for i in range(len(data[0]))]


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


def expand_node_vanilla(agent, node, tree):
    qs = node["qs"]
    policies = agent.policies

    qs_pi, _, _, G = step(agent, qs, policies)
    q_pi = nn.softmax(jnp.array(G), axis=0)

    for idx in range(len(policies)):
        policy_node = {
            "policy": policies[idx],
            "q_pi": q_pi[idx],
            "qs": jtu.tree_map(lambda x: x[idx, ...], qs_pi),
            "G": G[idx],
            "parent": node,
            "children": [],
            "n": node["n"] + 1,
        }
        node["children"].append(policy_node)
        tree.append(policy_node)

    # TODO update G of parents

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

    print(policies)
    qs_pi = agent.update_empirical_prior(qs, policies)
    qo_pi = jtu.tree_map(lambda a, q: a @ q, agent.A, qs_next)

    info_gain = compute_info_gain(qs_pi, qo_pi, agent.A, agent.A_dependencies)
    utility = compute_expected_utility(qo_pi, agent.C)
    G = utility + info_gain
    q_pi = nn.softmax(G * gamma)

    node["policies"] = policies
    node["q_pi"] = q_pi
    node["qs_pi"] = qs_pi
    node["qo_pi"] = qo_pi
    node["G"] = jnp.dot(q_pi, G)
    node["children"] = []

    # expand the policies and observations of this node
    ordered = jnp.argsort(q_pi)[::-1]
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
            "policy": policies[idx],
            "q_pi": q_pi[idx],
            "G": G[idx],
            "parent": node,
            "children": [],
            "n": node["n"] + 1,
        }
        node["children"].append(policy_node)
        tree.append(policy_node)

        if idx in policies_to_consider:
            # branch over possible observations
            qo_next = qo_pi[idx][0]
            for k in itertools.product(*[range(s.shape[0]) for s in qo_next]):
                prob = 1.0
                for i in range(len(k)):
                    prob *= qo_pi[idx][0][i][k[i]]

                # ignore low probability observations in the search tree
                if prob < observation_prune_threshold:
                    continue

                qo_one_hot = []
                for i in range(len(qo_one_hot)):
                    qo_one_hot = jnp.zeros(qo_next[i].shape[0])
                    qo_one_hot = qo_one_hot.at[k[i]].set(1.0)

                qs_next = agent.infer_states(qs_pi[idx], qo_one_hot)

                observation_node = {
                    "observation": qo_one_hot,
                    "prob": prob,
                    "qs": qs_next,
                    "G": 1e-10,
                    "parent": policy_node,
                    "children": [],
                    "n": node["n"] + 1,
                }
                policy_node["children"].append(observation_node)
                tree.append(observation_node)

        # TODO update Gs of parents

    return tree
