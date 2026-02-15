"""Smoke tests for the fast TMaze recoverability diagnostics."""

import math

from examples.model_fitting.tmaze_recoverability import RecoverabilityConfig, run_recoverability


def _small_cfg(parameterization: str) -> RecoverabilityConfig:
    return RecoverabilityConfig(
        parameterization=parameterization,
        seed=3,
        num_agents=5,
        num_blocks=12,
        num_trials=5,
        svi_steps=40,
        num_particles=1,
        num_samples=16,
    )


def test_tmaze_recoverability_three_param_smoke():
    results = run_recoverability(_small_cfg("three_param"))

    assert results["num_params"] == 3
    assert len(results["corr_latent"]) == 3
    assert len(results["true_reward_probability"]) == 5
    assert len(results["inferred_reward_probability"]) == 5
    assert 0.0 <= results["bimodality_score_reward_probability"] <= 1.0
    assert math.isfinite(results["corr_reward_probability"])


def test_tmaze_recoverability_reward_only_smoke():
    results = run_recoverability(_small_cfg("reward_only"))

    assert results["num_params"] == 1
    assert len(results["corr_latent"]) == 1
    assert len(results["true_reward_probability"]) == 5
    assert len(results["inferred_reward_probability"]) == 5
    assert 0.0 <= results["bimodality_score_reward_probability"] <= 1.0
    assert math.isfinite(results["corr_reward_probability"])
