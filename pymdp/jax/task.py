# Task environmnet
from typing import Optional, List, Dict
from jaxtyping import Array, PRNGKeyArray
from functools import partial

from equinox import Module, field, tree_at
from jax import vmap, random as jr, tree_util as jtu
import jax.numpy as jnp

def select_probs(positions, matrix, dependency_list, actions=None):
    args = tuple(p for i, p in enumerate(positions) if i in dependency_list)
    args += () if actions is None else (actions,)

    return matrix[..., *args]

def cat_sample(key, p):
    a = jnp.arange(p.shape[-1])
    if p.ndim > 1:
        choice = lambda key, p: jr.choice(key, a, p=p)
        keys = jr.split(key, len(p))
        return vmap(choice)(keys, p)

    return jr.choice(key, a, p=p)

class PyMDPEnv(Module):
    params: Dict
    states: List[List[Array]]
    dependencies: Dict = field(static=True)

    def __init__(
        self, params: Dict, dependencies: Dict, init_state: List[Array] = None
    ):
        self.params = params
        self.dependencies = dependencies

        if init_state is None:
            init_state = jtu.tree_map(lambda x: jnp.argmax(x, -1), self.params["D"])

        self.states = [init_state]

    def reset(self, key: Optional[PRNGKeyArray] = None):
        if key is None:
            states = [self.states[0]]
        else:
            probs = self.params["D"]
            keys = list(jr.split(key, len(probs)))
            states = [jtu.tree_map(cat_sample, keys, probs)]

        return tree_at(lambda x: x.states, self, states)

    @vmap
    def step(self, key: PRNGKeyArray, actions: Optional[Array] = None):
        # return a list of random observations and states
        key_state, key_obs = jr.split(key)
        states = self.states
        if actions is not None:
            actions = list(actions)
            _select_probs = partial(select_probs, states[-1])
            state_probs = jtu.tree_map(
                _select_probs, self.params["B"], self.dependencies["B"], actions
            )

            keys = list(jr.split(key_state, len(state_probs)))
            new_states = jtu.tree_map(cat_sample, keys, state_probs)
        else:
            new_states = states[-1]

        _select_probs = partial(select_probs, new_states)
        obs_probs = jtu.tree_map(
            _select_probs, self.params["A"], self.dependencies["A"]
        )

        keys = list(jr.split(key_obs, len(obs_probs)))
        new_obs = jtu.tree_map(cat_sample, keys, obs_probs)

        return new_obs, tree_at(lambda x: (x.states), self, [new_states])