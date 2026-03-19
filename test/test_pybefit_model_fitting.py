"""Smoke tests for pybefit-based model fitting integration."""

import pytest
from jax import nn, random as jr
from jax import lax
from jax import numpy as jnp
from jax import tree_util as jtu

from pymdp.agent import Agent
from pymdp.envs import TMaze


pybefit = pytest.importorskip("pybefit")
from pybefit.inference import Normal, NumpyroModel  # noqa: E402
from pybefit.inference.numpyro.likelihoods import pymdp_likelihood  # noqa: E402
from numpyro.infer import Predictive  # noqa: E402


def _build_tmaze_agent_transform(task):
    """Build the notebook-style parameter transform used for pybefit fitting."""

    def transform(z):
        num_agents = z.shape[0]

        reward_prob = nn.sigmoid(z[..., 0])
        lam = nn.softplus(z[..., 1])
        d = nn.sigmoid(z[..., 2])

        A = lax.stop_gradient(task.A)
        A = jtu.tree_map(lambda x: jnp.broadcast_to(x, (num_agents,) + x.shape), A)
        B = lax.stop_gradient(task.B)
        B = jtu.tree_map(lambda x: jnp.broadcast_to(x, (num_agents,) + x.shape), B)

        one_minus = 1.0 - reward_prob
        reward_left = jnp.stack([reward_prob, one_minus], -1)
        punish_left = jnp.stack([one_minus, reward_prob], -1)
        reward_right = jnp.stack([one_minus, reward_prob], -1)
        punish_right = jnp.stack([reward_prob, one_minus], -1)
        zeros = jnp.zeros_like(reward_left)

        side = jnp.broadcast_to(
            jnp.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]), (num_agents, 3, 2)
        )
        left_col = jnp.stack([zeros, reward_left, punish_left], axis=-2)
        right_col = jnp.stack([zeros, reward_right, punish_right], axis=-2)
        A[1] = jnp.stack([side, left_col, right_col, side, side], axis=-2)

        C = [
            jnp.zeros((num_agents, A[0].shape[1])),
            jnp.expand_dims(lam, -1) * jnp.array([0.0, 1.0, -1.0]),
            jnp.zeros((num_agents, A[2].shape[1])),
        ]
        D = [
            jnp.zeros((num_agents, B[0].shape[1])).at[:, 0].set(1.0),
            jnp.stack([d, 1.0 - d], -1),
        ]

        return Agent(
            A,
            B,
            C=C,
            D=D,
            policy_len=2,
            A_dependencies=task.A_dependencies,
            B_dependencies=task.B_dependencies,
            batch_size=num_agents,
        )

    return transform


def test_pybefit_tmaze_predictive_smoke():
    """
    Run a minimal pybefit+pymdp predictive pass used in docs model-fitting examples.

    This is intended to catch API drift between pybefit and pymdp early.
    """

    num_blocks = 2
    num_trials = 2
    num_agents = 1
    num_params = 3
    key = jr.PRNGKey(0)

    task = TMaze(
        reward_probability=0.98,
        punishment_probability=0.0,
        cue_validity=1.0,
        reward_condition=None,
        dependent_outcomes=True,
    )
    prior = Normal(num_params, num_agents, backend="numpyro")
    transform = _build_tmaze_agent_transform(task)

    opts_task = {
        "task": task,
        "num_blocks": num_blocks,
        "num_trials": num_trials,
        "num_agents": num_agents,
    }
    opts_model = {"prior": {}, "transform": {}, "likelihood": opts_task}

    model = NumpyroModel(prior, transform, pymdp_likelihood, opts=opts_model)
    pred = Predictive(model, num_samples=1)
    key, subkey = jr.split(key)
    samples = pred(subkey)

    assert "outcomes" in samples
    observations = samples["outcomes"]
    assert len(observations) == 3
    # Observations include the initial observation plus one per trial.
    assert observations[0].shape == (1, num_blocks, num_agents, num_trials + 1)
