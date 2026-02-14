from functools import partial
from abc import ABC, abstractmethod
from typing import Any, Sequence

import jax.numpy as jnp
from jax import vmap, jit, random as jr, tree_util as jtu
from jaxtyping import Array

from pymdp.distribution import Distribution, get_dependencies


def _float_to_int_index(x: Array) -> Array:
    # converting float to integer for array indexing while preserving the og data structure for gradient computation    
    return jnp.asarray(x, jnp.int32)

def select_probs(
    positions: Sequence[Array],
    matrix: Array,
    dependency_list: Sequence[int],
    actions: Array | None = None,
) -> Array:
    # creating integer indices from float state positions for the positions specified in dependency_list
    index_args = tuple(_float_to_int_index(positions[i]) for i in dependency_list)
    if actions is not None:
        index_args += (_float_to_int_index(actions),)
    return matrix[(..., *index_args)]


def cat_sample(key: Array, p: Array) -> Array:
    a = jnp.arange(p.shape[-1], dtype=jnp.float32)
    if p.ndim > 1:
        choice = lambda key, p: jr.choice(key, a, p=p)
        keys = jr.split(key, len(p))
        # print(keys.shape)
        return vmap(choice)(keys, p)

    return jr.choice(key, a, p=p)


def make(
    A: Sequence[Array] | Sequence[Distribution],
    B: Sequence[Array] | Sequence[Distribution],
    D: Sequence[Array] | Sequence[Distribution],
    A_dependencies: list[list[int]] | None = None,
    B_dependencies: list[list[int]] | None = None,
    make_env_params: bool = False,
    **kwargs: Any,
) -> tuple["PymdpEnv", dict[str, list[Array]] | None]:
    """
    Convenience factory to construct a `PymdpEnv`.

    Parameters
    ----------
    A, B, D: sequence of arrays or Distribution
        Likelihoods, transitions, and priors describing the POMDP. Each entry
        corresponds to a modality (A) or state factor (B, D). Entries may be
        raw arrays or `Distribution` instances.
    A_dependencies, B_dependencies: list, optional
        Explicit dependency structures. If omitted, they are inferred.
    make_env_params: bool
        If True, also return a dict of env_params (with Distribution instances
        converted to their `.data`). Otherwise, env_params is None. Returned
        env_params match the input shapes (shapes of `A`, `B`, and `D`) and are 
        not broadcast; for broadcasting single-batch environmental parameters to a larger batch size,
        call `env.generate_env_params(batch_size=...)`  on the `env` that is returned by this function.
    kwargs: dict
        Passed through to `PymdpEnv`.

    Returns
    -------
    env: PymdpEnv
    env_params: dict | None
    """
    env = PymdpEnv(A=A, B=B, D=D, A_dependencies=A_dependencies, B_dependencies=B_dependencies, **kwargs)
    if not make_env_params:
        return env, None

    def _to_arrays(params: Sequence[Array] | Sequence[Distribution] | None) -> list[Array] | None:
        if params is None:
            return None
        return [jnp.array(p.data) if isinstance(p, Distribution) else p for p in params]

    env_params = {"A": _to_arrays(A), "B": _to_arrays(B), "D": _to_arrays(D)}

    return env, env_params

class Env(ABC):

    @abstractmethod
    def reset(
        self,
        key: Array,
        state: list[Array] | None = None,
        env_params: dict[str, list[Array]] | None = None,
    ) -> tuple[list[Array], list[Array]]:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        key: Array,
        state: list[Array],
        action: Array | None,
        env_params: dict[str, list[Array]] | None = None,
    ) -> tuple[list[Array], list[Array]]:
        raise NotImplementedError

    def generate_env_params(
        self, key: Array | None = None, batch_size: int | None = None
    ) -> dict[str, list[Array]] | None:
        return None

class PymdpEnv(Env):

    def __init__(
        self,
        A: Sequence[Array] | Sequence[Distribution] | None = None,
        B: Sequence[Array] | Sequence[Distribution] | None = None,
        D: Sequence[Array] | Sequence[Distribution] | None = None,
        A_dependencies: list[list[int]] | None = None,
        B_dependencies: list[list[int]] | None = None,
        **kwargs: Any,
    ) -> None:
        if A_dependencies is not None:
            self.A_dependencies = A_dependencies
        elif A is not None and B is not None:
            if isinstance(A[0], Distribution) and isinstance(B[0], Distribution):
                self.A_dependencies, _ = get_dependencies(A, B)
            else:
                self.A_dependencies = [list(range(len(B))) for _ in range(len(A))]
        else:
            raise ValueError("Need to provide A and B or A_dependencies")

        if B_dependencies is not None:
            self.B_dependencies = B_dependencies
        elif A is not None and B is not None:
            if isinstance(A[0], Distribution) and isinstance(B[0], Distribution):
                _, self.B_dependencies = get_dependencies(A, B)
            else:
                self.B_dependencies = [[f] for f in range(len(B))]
        else:
            raise ValueError("Need to provide A and B or B_dependencies")

        if A is not None:
            self.A = [jnp.array(a.data) if isinstance(a, Distribution) else a for a in A]
        else:
            self.A = None

        if B is not None:
            self.B = [jnp.array(b.data) if isinstance(b, Distribution) else b for b in B]
        else:
            self.B = None

        if D is not None:
            self.D = [jnp.array(d.data) if isinstance(d, Distribution) else d for d in D]
        else:
            self.D = None

    def generate_env_params(
        self, key: Array | None = None, batch_size: int | None = None
    ) -> dict[str, list[Array]]:
        env_params = {"A": self.A, "B": self.B, "D": self.D}
        if batch_size is None:
            return env_params

        expand_to_batch = lambda x: jnp.broadcast_to(jnp.asarray(x), (batch_size,) + x.shape)
        return jtu.tree_map(expand_to_batch, {"A": self.A, "B": self.B, "D": self.D})
    
    @partial(jit, static_argnums=(0,))
    def reset(
        self,
        key: Array,
        state: list[Array] | None = None,
        env_params: dict[str, list[Array]] | None = None,
    ) -> tuple[list[Array], list[Array]]:
        if state is None:
            probs = env_params["D"] if env_params is not None else self.D
            keys = list(jr.split(key, len(probs) + 1))
            key = keys[0]
            state = jtu.tree_map(cat_sample, keys[1:], probs)
        obs = self._sample_obs(key, state, env_params)
        return obs, state

    @partial(jit, static_argnums=(0,))
    def step(
        self,
        key: Array,
        state: list[Array],
        action: Array | None,
        env_params: dict[str, list[Array]] | None = None,
    ) -> tuple[list[Array], list[Array]]:
        key_state, key_obs = jr.split(key)
        if action is not None:
            action = list(action)
            _select_probs = partial(select_probs, state)
            B = env_params["B"] if env_params is not None else self.B
            state_probs = jtu.tree_map(_select_probs, B, self.B_dependencies, action)

            keys = list(jr.split(key_state, len(state_probs)))
            new_state = jtu.tree_map(cat_sample, keys, state_probs)
        else:
            new_state = state

        new_obs = self._sample_obs(key_obs, new_state, env_params)

        return new_obs, new_state

    def _sample_obs(
        self, key: Array, state: list[Array], env_params: dict[str, list[Array]] | None
    ) -> list[Array]:
        _select_probs = partial(select_probs, state)
        A = env_params["A"] if env_params is not None else self.A
        obs_probs = jtu.tree_map(_select_probs, A, self.A_dependencies)

        keys = list(jr.split(key, len(obs_probs)))
        new_obs = jtu.tree_map(cat_sample, keys, obs_probs)
        new_obs = jtu.tree_map(lambda x: jnp.expand_dims(x, -1), new_obs)
        return new_obs
