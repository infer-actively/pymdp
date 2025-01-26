from typing import Optional, List, Dict
from jaxtyping import Array, PRNGKeyArray
from functools import partial

from equinox import Module, field, tree_at
from jax import vmap, random as jr, tree_util as jtu
import jax.numpy as jnp

def _float_to_int_index(x):
    # converting float to integer for array indexing while preserving the og data structure for gradient computation    
    return jnp.asarray(x, jnp.int32)

def select_probs(positions, matrix, dependency_list, actions=None):
    # creating integer indices from float state positions for the positions specified in dependency_list
    index_args = tuple(_float_to_int_index(p) for i, p in enumerate(positions) 
                      if i in dependency_list)
    if actions is not None:
        index_args += (_float_to_int_index(actions),)
    return matrix[..., *index_args]


def cat_sample(key, p):
    a = jnp.arange(p.shape[-1], dtype=jnp.float32)
    if p.ndim > 1:
        choice = lambda key, p: jr.choice(key, a, p=p)
        keys = jr.split(key, len(p))
        # print(keys.shape)
        return vmap(choice)(keys, p)

    return jr.choice(key, a, p=p)


class Env(Module):
    params: Dict
    state: List[Array]
    current_obs: List[Array]
    dependencies: Dict = field(static=True)

    def __init__(self, params: Dict, dependencies: Dict):
        self.params = params
        self.dependencies = dependencies

        self.state = jtu.tree_map(lambda x: jnp.zeros([x.shape[0]]), self.params["D"])
        self.current_obs = jtu.tree_map(lambda x: jnp.zeros([x.shape[0], x.shape[1]]), self.params["A"])

    @vmap
    def reset(self, key: Optional[PRNGKeyArray], state: Optional[List[Array]] = None):
        if state is None:
            probs = self.params["D"]
            keys = list(jr.split(key, len(probs) + 1))
            key = keys[0]
            state = jtu.tree_map(cat_sample, keys[1:], probs)

        env = tree_at(lambda x: x.state, self, state)

        new_obs = self._sample_obs(key, state)
        env = tree_at(lambda x: x.current_obs, env, new_obs)
        return new_obs, env

    def render(self, mode="human"):
        """

        Returns
        ----
        if mode == "human":
            returns None, renders the environment using MPL inside the function
        elif mode == "rgb_array":
            A (H, W, 3) uint8 jax.numpy array, with values between 0 and 255
        """
        pass

    @vmap
    def step(self, rng_key: PRNGKeyArray, actions: Optional[Array] = None):
        # return a list of random observations and states
        key_state, key_obs = jr.split(rng_key)
        state = self.state
        if actions is not None:
            actions = list(actions)
            _select_probs = partial(select_probs, state)
            state_probs = jtu.tree_map(_select_probs, self.params["B"], self.dependencies["B"], actions)

            keys = list(jr.split(key_state, len(state_probs)))
            new_state = jtu.tree_map(cat_sample, keys, state_probs)
        else:
            new_state = state

        new_obs = self._sample_obs(key_obs, new_state)

        env = tree_at(lambda x: (x.state), self, new_state)
        env = tree_at(lambda x: x.current_obs, env, new_obs)
        return new_obs, env

    def _sample_obs(self, key, state):
        _select_probs = partial(select_probs, state)
        obs_probs = jtu.tree_map(_select_probs, self.params["A"], self.dependencies["A"])

        keys = list(jr.split(key, len(obs_probs)))
        new_obs = jtu.tree_map(cat_sample, keys, obs_probs)
        new_obs = jtu.tree_map(lambda x: jnp.expand_dims(x, -1), new_obs)
        return new_obs
