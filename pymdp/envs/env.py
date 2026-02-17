"""Environment interfaces and POMDP-backed environment utilities.

This module provides:
- `Env`: an abstract JAX-compatible environment interface used by `rollout()`,
- `PymdpEnv`: a concrete environment driven by categorical `A`, `B`, and `D`,
- `make(...)`: a convenience constructor for `PymdpEnv` and optional params.
"""

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Sequence

import jax.numpy as jnp
from jax import jit, random as jr, tree_util as jtu, vmap
from jaxtyping import Array

from pymdp.distribution import Distribution, get_dependencies


def _float_to_int_index(x: Array) -> Array:
    """Cast index arrays to int32 for safe advanced indexing.

    Parameters
    ----------
    x : Array
        Index leaf, potentially represented in floating type.

    Returns
    -------
    Array
        Integer index array (`int32`) with the same structure as `x`.
    """
    return jnp.asarray(x, jnp.int32)


def select_probs(
    positions: Sequence[Array],
    matrix: Array,
    dependency_list: Sequence[int],
    actions: Array | None = None,
) -> Array:
    """Select conditional probabilities from a factorized tensor.

    Parameters
    ----------
    positions : Sequence[Array]
        Current hidden-state indices for all factors.
    matrix : Array
        Likelihood or transition tensor to index.
    dependency_list : Sequence[int]
        Indices of factors used to index `matrix` lagging dimensions.
    actions : Array | None, optional
        Optional action index (or action indices) used to select the final
        action axis in transition tensors.

    Returns
    -------
    Array
        Selected conditional probability vector(s).
    """
    index_args = tuple(_float_to_int_index(positions[i]) for i in dependency_list)
    if actions is not None:
        index_args += (_float_to_int_index(actions),)
    return matrix[(..., *index_args)]


def cat_sample(key: Array, p: Array) -> Array:
    """Sample from one or more categorical distributions.

    Parameters
    ----------
    key : Array
        JAX PRNG key.
    p : Array
        Probability vector or batch of probability vectors on the last axis.

    Returns
    -------
    Array
        Sampled category index/indices as floating values.
    """
    a = jnp.arange(p.shape[-1], dtype=jnp.float32)
    if p.ndim > 1:
        choice = lambda key, p: jr.choice(key, a, p=p)
        keys = jr.split(key, len(p))
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
    """Construct a `PymdpEnv` (and optionally environment parameters).

    Parameters
    ----------
    A : sequence[Array] | sequence[Distribution]
        Observation likelihood tensors, one per observation modality.
    B : sequence[Array] | sequence[Distribution]
        Transition tensors, one per hidden-state factor.
    D : sequence[Array] | sequence[Distribution]
        Initial-state priors, one per hidden-state factor.
    A_dependencies : list[list[int]] | None, optional
        Explicit modality-to-state dependencies. If `None`, dependencies are
        inferred when possible.
    B_dependencies : list[list[int]] | None, optional
        Explicit state-transition dependencies. If `None`, dependencies are
        inferred when possible.
    make_env_params : bool, default=False
        If `True`, also return `env_params={"A": ..., "B": ..., "D": ...}`
        with `Distribution` entries converted to dense arrays.
    **kwargs : Any
        Additional keyword arguments forwarded to `PymdpEnv`.

    Returns
    -------
    tuple[PymdpEnv, dict[str, list[Array]] | None]
        Constructed environment and optional unbatched environment parameters.
        To broadcast parameters to a larger batch, call
        `env.generate_env_params(batch_size=...)`.
    """
    env = PymdpEnv(
        A=A,
        B=B,
        D=D,
        A_dependencies=A_dependencies,
        B_dependencies=B_dependencies,
        **kwargs,
    )
    if not make_env_params:
        return env, None

    def _to_arrays(params: Sequence[Array] | Sequence[Distribution] | None) -> list[Array] | None:
        if params is None:
            return None
        return [jnp.array(p.data) if isinstance(p, Distribution) else p for p in params]

    env_params = {"A": _to_arrays(A), "B": _to_arrays(B), "D": _to_arrays(D)}

    return env, env_params


class Env(ABC):
    """Abstract JAX-compatible environment interface used by `rollout()`."""

    @abstractmethod
    def reset(
        self,
        key: Array,
        state: list[Array] | None = None,
        env_params: dict[str, list[Array]] | None = None,
    ) -> tuple[list[Array], list[Array]]:
        """Reset environment state and return initial observation/state.

        Parameters
        ----------
        key : Array
            JAX PRNG key.
        state : list[Array] | None, optional
            Optional explicit initial hidden state.
        env_params : dict[str, list[Array]] | None, optional
            Optional runtime override for environment parameters.

        Returns
        -------
        tuple[list[Array], list[Array]]
            Initial observations and hidden state.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        key: Array,
        state: list[Array],
        action: Array | None,
        env_params: dict[str, list[Array]] | None = None,
    ) -> tuple[list[Array], list[Array]]:
        """Advance one environment step and return new observation/state.

        Parameters
        ----------
        key : Array
            JAX PRNG key.
        state : list[Array]
            Current hidden state.
        action : Array | None
            Action sampled by the agent. `None` can be used for no-op updates.
        env_params : dict[str, list[Array]] | None, optional
            Optional runtime override for environment parameters.

        Returns
        -------
        tuple[list[Array], list[Array]]
            Next observations and hidden state.
        """
        raise NotImplementedError

    def generate_env_params(
        self, key: Array | None = None, batch_size: int | None = None
    ) -> dict[str, list[Array]] | None:
        """Generate optional environment parameter pytrees.

        Parameters
        ----------
        key : Array | None, optional
            Optional JAX PRNG key (unused by default implementation).
        batch_size : int | None, optional
            Optional batch size for parameter generation.

        Returns
        -------
        dict[str, list[Array]] | None
            Environment parameters or `None` if not implemented.
        """
        return None


class PymdpEnv(Env):
    """Environment whose dynamics are defined by categorical `A`, `B`, and `D`.

    `PymdpEnv` is useful when the environment is isomorphic to a discrete POMDP
    generative process:
    - `A[m]`: observation likelihoods per modality,
    - `B[f]`: transitions per hidden-state factor,
    - `D[f]`: initial-state priors per hidden-state factor.
    """

    def __init__(
        self,
        A: Sequence[Array] | Sequence[Distribution] | None = None,
        B: Sequence[Array] | Sequence[Distribution] | None = None,
        D: Sequence[Array] | Sequence[Distribution] | None = None,
        A_dependencies: list[list[int]] | None = None,
        B_dependencies: list[list[int]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize `PymdpEnv`.

        Parameters
        ----------
        A : sequence[Array] | sequence[Distribution] | None, optional
            Observation likelihood tensors.
        B : sequence[Array] | sequence[Distribution] | None, optional
            Transition tensors.
        D : sequence[Array] | sequence[Distribution] | None, optional
            Initial-state priors.
        A_dependencies : list[list[int]] | None, optional
            Modality-to-state dependencies for `A`.
        B_dependencies : list[list[int]] | None, optional
            State-to-state dependencies for `B`.
        **kwargs : Any
            Accepted for forward compatibility.
        """
        del kwargs

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
        """Return default environment params, optionally broadcast to batch.

        Parameters
        ----------
        key : Array | None, optional
            Optional JAX PRNG key (unused).
        batch_size : int | None, optional
            If provided, broadcast each parameter leaf with leading shape
            `(batch_size, ...)`.

        Returns
        -------
        dict[str, list[Array]]
            Dictionary with keys `"A"`, `"B"`, and `"D"`.
        """
        del key

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
        """Reset state and emit an initial observation sample.

        If `state` is omitted, states are sampled from `D`.
        """
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
        """Advance the process by one timestep.

        If `action` is provided, next hidden states are sampled from `B`.
        Observations are then sampled from `A` conditioned on the new state.
        """
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
        """Sample all observation modalities conditioned on hidden state."""
        _select_probs = partial(select_probs, state)
        A = env_params["A"] if env_params is not None else self.A
        obs_probs = jtu.tree_map(_select_probs, A, self.A_dependencies)

        keys = list(jr.split(key, len(obs_probs)))
        new_obs = jtu.tree_map(cat_sample, keys, obs_probs)
        new_obs = jtu.tree_map(lambda x: jnp.expand_dims(x, -1), new_obs)
        return new_obs
