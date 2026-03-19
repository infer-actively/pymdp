"""Fast recoverability diagnostics for pybefit + pymdp TMaze fitting.

This module provides a lightweight alternative to running the full
`fitting_with_pybefit.ipynb` notebook when evaluating identifiability.
It focuses on synthetic recovery sweeps with fixed per-agent latent grids,
then reports parameter correlation and a simple bimodality diagnostic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
from jax import lax, nn
from jax import random as jr
from numpyro.infer import Predictive

from pybefit.inference import (
    NumpyroGuide,
    NumpyroModel,
    Normal,
    NormalPosterior,
    default_dict_numpyro_svi,
    run_svi,
)
from pybefit.inference.numpyro.likelihoods import pymdp_likelihood as likelihood

from pymdp.agent import Agent
from pymdp.envs import TMaze


@dataclass(frozen=True)
class RecoverabilityConfig:
    """Configuration for synthetic TMaze recoverability diagnostics."""

    parameterization: str = "three_param"
    seed: int = 0
    num_agents: int = 9
    num_blocks: int = 80
    num_trials: int = 5
    svi_steps: int = 300
    num_particles: int = 5
    num_samples: int = 80
    reward_probability: float = 0.98
    punishment_probability: float = 0.0
    cue_validity: float = 1.0
    dependent_outcomes: bool = True
    save_plot: str | None = None
    save_json: str | None = None


def _assert_parameterization(parameterization: str) -> None:
    valid = {"three_param", "reward_only"}
    if parameterization not in valid:
        raise ValueError(f"parameterization must be one of {sorted(valid)}, got {parameterization!r}")


def _build_task(cfg: RecoverabilityConfig) -> TMaze:
    return TMaze(
        reward_probability=cfg.reward_probability,
        punishment_probability=cfg.punishment_probability,
        cue_validity=cfg.cue_validity,
        reward_condition=None,
        dependent_outcomes=cfg.dependent_outcomes,
    )


def _latent_grid(cfg: RecoverabilityConfig, num_params: int) -> jnp.ndarray:
    """Construct deterministic per-agent latent values for reproducible recovery checks."""

    base = jnp.linspace(-2.5, 2.5, cfg.num_agents)
    if num_params == 1:
        return base[:, None]

    key = jr.PRNGKey(cfg.seed + 13)
    key_1, key_2 = jr.split(key)
    z_1 = 0.8 * jr.normal(key_1, (cfg.num_agents,))
    z_2 = 0.8 * jr.normal(key_2, (cfg.num_agents,))
    return jnp.stack([base, z_1, z_2], axis=-1)


def _build_three_param_transform(task: TMaze):
    """Notebook-aligned transform with free reward probability, lambda, and initial reward-state prior."""

    def transform(z: jnp.ndarray) -> Agent:
        num_agents, _ = z.shape
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

        side = jnp.broadcast_to(jnp.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]), (num_agents, 3, 2))
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


def _build_reward_only_transform(task: TMaze):
    """Constrained transform with only reward probability free (lambda and D fixed).

    This removes the strongest confounds in the 3-parameter notebook setup and
    provides a cleaner recoverability baseline.
    """

    def transform(z: jnp.ndarray) -> Agent:
        num_agents, _ = z.shape
        reward_prob = nn.sigmoid(z[..., 0])
        lam = jnp.full((num_agents,), 1.2)

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

        side = jnp.broadcast_to(jnp.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]), (num_agents, 3, 2))
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
            jnp.full((num_agents, 2), 0.5),
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


def _build_model_and_truth(cfg: RecoverabilityConfig) -> tuple[NumpyroModel, jnp.ndarray, int]:
    _assert_parameterization(cfg.parameterization)
    task = _build_task(cfg)

    if cfg.parameterization == "three_param":
        num_params = 3
        transform = _build_three_param_transform(task)
    else:
        num_params = 1
        transform = _build_reward_only_transform(task)

    prior = Normal(num_params, cfg.num_agents, backend="numpyro")
    opts_task = {
        "task": task,
        "num_blocks": cfg.num_blocks,
        "num_trials": cfg.num_trials,
        "num_agents": cfg.num_agents,
    }
    model = NumpyroModel(prior, transform, likelihood, opts={"prior": {}, "transform": {}, "likelihood": opts_task})
    z_true = _latent_grid(cfg, num_params)
    return model, z_true, num_params


def _simulate_measurements(model: NumpyroModel, z_true: jnp.ndarray, seed: int) -> dict[str, Any]:
    pred = Predictive(model, posterior_samples={"z": z_true[None, ...]}, return_sites=["z", "outcomes", "multiactions"])
    samples = pred(jr.PRNGKey(seed))
    observations = samples["outcomes"]
    return {
        "samples": samples,
        "measurements": {
            "outcomes": [obs[0] for obs in observations],
            "multiactions": samples["multiactions"][0],
        },
    }


def _fit_svi(
    model: NumpyroModel,
    measurements: dict[str, Any],
    num_params: int,
    cfg: RecoverabilityConfig,
) -> dict[str, Any]:
    posterior = NumpyroGuide(NormalPosterior(num_params, cfg.num_agents, backend="numpyro"))
    opts_svi = default_dict_numpyro_svi | {
        "seed": cfg.seed,
        "iter_steps": cfg.svi_steps,
        "elbo_kwargs": {"num_particles": cfg.num_particles, "max_plate_nesting": 1},
        "svi_kwargs": {"progress_bar": False, "stable_update": True},
        "sample_kwargs": {"num_samples": cfg.num_samples},
    }
    svi_samples, _, _ = run_svi(model, posterior, measurements, opts=opts_svi)
    return svi_samples


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _bimodality_score(probabilities: np.ndarray) -> float:
    """Simple threshold-bimodality score in [0, 1], higher means more mass near extremes."""

    low_or_high = np.mean((probabilities < 0.2) | (probabilities > 0.8))
    middle = np.mean((probabilities >= 0.35) & (probabilities <= 0.65))
    score = low_or_high - middle
    return float(np.clip(score, 0.0, 1.0))


def run_recoverability(cfg: RecoverabilityConfig) -> dict[str, Any]:
    """Run one synthetic recoverability sweep and return diagnostics."""

    model, z_true, num_params = _build_model_and_truth(cfg)
    simulation = _simulate_measurements(model, z_true, cfg.seed + 1)
    svi_samples = _fit_svi(model, simulation["measurements"], num_params, cfg)

    true_z = np.array(z_true)
    inferred_z = np.array(svi_samples["z"].mean(0))
    latent_corr = [_pearson(true_z[:, idx], inferred_z[:, idx]) for idx in range(num_params)]

    true_reward_prob = 1.0 / (1.0 + np.exp(-true_z[:, 0]))
    inferred_reward_prob = 1.0 / (1.0 + np.exp(-inferred_z[:, 0]))

    results = {
        "config": asdict(cfg),
        "num_params": num_params,
        "corr_latent": latent_corr,
        "corr_reward_probability": _pearson(true_reward_prob, inferred_reward_prob),
        "bimodality_score_reward_probability": _bimodality_score(inferred_reward_prob),
        "true_z": true_z.tolist(),
        "inferred_z": inferred_z.tolist(),
        "true_reward_probability": true_reward_prob.tolist(),
        "inferred_reward_probability": inferred_reward_prob.tolist(),
    }
    return results


def _save_scatter(results: dict[str, Any], path: Path) -> None:
    true_prob = np.array(results["true_reward_probability"])
    inferred_prob = np.array(results["inferred_reward_probability"])

    plt.figure(figsize=(7, 5))
    plt.scatter(true_prob, inferred_prob, label="reward probability")
    lower = float(min(true_prob.min(), inferred_prob.min()))
    upper = float(max(true_prob.max(), inferred_prob.max()))
    plt.plot((lower, upper), (lower, upper), "k--", linewidth=1.0)
    plt.xlabel("true value")
    plt.ylabel("posterior mean (SVI)")
    plt.title("TMaze Recoverability")
    plt.legend(loc="best")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def _to_jsonable(results: dict[str, Any]) -> str:
    return json.dumps(results, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameterization", choices=["three_param", "reward_only"], default="three_param")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-agents", type=int, default=9)
    parser.add_argument("--num-blocks", type=int, default=80)
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--svi-steps", type=int, default=300)
    parser.add_argument("--num-particles", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=80)
    parser.add_argument("--save-plot", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RecoverabilityConfig(
        parameterization=args.parameterization,
        seed=args.seed,
        num_agents=args.num_agents,
        num_blocks=args.num_blocks,
        num_trials=args.num_trials,
        svi_steps=args.svi_steps,
        num_particles=args.num_particles,
        num_samples=args.num_samples,
        save_plot=args.save_plot,
        save_json=args.save_json,
    )
    results = run_recoverability(cfg)

    print(f"parameterization={cfg.parameterization}")
    print(f"corr_reward_probability={results['corr_reward_probability']:.3f}")
    print(f"bimodality_score_reward_probability={results['bimodality_score_reward_probability']:.3f}")
    for idx, corr in enumerate(results["corr_latent"]):
        print(f"corr_latent_{idx}={corr:.3f}")

    if cfg.save_plot:
        _save_scatter(results, Path(cfg.save_plot))
        print(f"saved_plot={cfg.save_plot}")

    if cfg.save_json:
        out_path = Path(cfg.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_to_jsonable(results))
        print(f"saved_json={cfg.save_json}")


if __name__ == "__main__":
    main()
