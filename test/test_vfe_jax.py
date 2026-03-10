#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from itertools import product

import numpy as np
from jax import grad, nn
from jax import numpy as jnp
from jax.scipy.special import digamma, gammaln

from pymdp import inference, maths
from pymdp.agent import Agent
from pymdp.legacy import maths as legacy_maths
from pymdp.legacy import utils as legacy_utils


def _manual_dirichlet_kl(q_dir: np.ndarray, p_dir: np.ndarray) -> float:
    q_sum = q_dir.sum(axis=0)
    p_sum = p_dir.sum(axis=0)
    digamma_q_sum = digamma(q_sum)[None, ...]
    kl = (
        gammaln(q_sum)
        - gammaln(p_sum)
        - gammaln(q_dir).sum(axis=0)
        + gammaln(p_dir).sum(axis=0)
        + ((q_dir - p_dir) * (digamma(q_dir) - digamma_q_sum)).sum(axis=0)
    )
    return float(np.asarray(kl).sum())


def _bruteforce_sequence_vfe(
    prior: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    obs_idx: np.ndarray,
    actions: np.ndarray,
) -> float:
    trajectories = np.array(list(product(range(prior.shape[0]), repeat=len(obs_idx))), dtype=int)
    log_joint = np.zeros((trajectories.shape[0],), dtype=float)

    for i, traj in enumerate(trajectories):
        log_prob = np.log(prior[traj[0]])
        log_prob += np.log(A[obs_idx[0], traj[0]])
        for t in range(1, len(obs_idx)):
            log_prob += np.log(B[traj[t], traj[t - 1], actions[t - 1]])
            log_prob += np.log(A[obs_idx[t], traj[t]])
        log_joint[i] = log_prob

    log_norm = np.logaddexp.reduce(log_joint)
    posterior = np.exp(log_joint - log_norm)
    return float(np.sum(posterior * (np.log(posterior) - log_joint)))


def _filtered_history_single_factor(
    method: str,
    A: list[jnp.ndarray],
    B: list[jnp.ndarray],
    prior: list[jnp.ndarray],
    obs: list[jnp.ndarray],
    actions: jnp.ndarray,
) -> list[jnp.ndarray]:
    qs_hist = None
    prior_t = prior
    for t in range(obs[0].shape[0]):
        qs_hist = inference.update_posterior_states(
            A=A,
            B=B,
            obs=[obs[0][t : t + 1]],
            past_actions=None,
            prior=prior_t,
            qs_hist=qs_hist,
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            method=method,
            distr_obs=True,
            inference_horizon=obs[0].shape[0],
        )
        if t < obs[0].shape[0] - 1:
            prior_t = [B[0][..., int(actions[t])] @ qs_hist[0][-1]]
    return qs_hist


class TestCanonicalVFE(unittest.TestCase):
    def test_calc_vfe_single_step_matches_legacy_full_model(self):
        prior = [
            jnp.array([0.55, 0.45]),
            jnp.array([0.30, 0.70]),
        ]
        qs = [
            jnp.array([0.25, 0.75]),
            jnp.array([0.60, 0.40]),
        ]
        obs = [jnp.array([0.0, 1.0, 0.0])]
        A = [
            jnp.array(
                [
                    [[0.80, 0.20], [0.30, 0.10]],
                    [[0.15, 0.30], [0.30, 0.20]],
                    [[0.05, 0.50], [0.40, 0.70]],
                ]
            )
        ]

        _, vfe_jax = maths.calc_vfe(
            qs,
            prior,
            obs=obs,
            A=A,
            A_dependencies=[[0, 1]],
        )

        qs_np = legacy_utils.obj_array_from_list([np.asarray(q) for q in qs])
        prior_np = legacy_utils.obj_array_from_list([np.asarray(p) for p in prior])
        prior_log_np = legacy_maths.spm_log_obj_array(prior_np)
        A_np = legacy_utils.obj_array_from_list([np.asarray(a) for a in A])
        obs_np = legacy_utils.obj_array_from_list([np.asarray(o) for o in obs])
        ll_np = legacy_maths.spm_log_single(
            legacy_maths.get_joint_likelihood(A_np, obs_np, [2, 2])
        )
        vfe_legacy = legacy_maths.calc_free_energy(
            qs_np,
            prior_log_np,
            n_factors=2,
            likelihood=ll_np,
        )

        np.testing.assert_allclose(
            np.asarray(vfe_jax),
            np.asarray(vfe_legacy).squeeze(),
            atol=1e-6,
        )

    def test_calc_vfe_sequence_matches_manual_terms_and_parameter_kls(self):
        prior = [jnp.array([0.65, 0.35])]
        qs = [jnp.array([[0.80, 0.20], [0.25, 0.75], [0.60, 0.40]])]
        obs = [jnp.array([[1.0, 0.0], [0.0, 1.0], [0.40, 0.60]])]
        A = [jnp.array([[0.90, 0.15], [0.10, 0.85]])]
        B = [
            jnp.array(
                [
                    [[0.85, 0.30], [0.20, 0.65]],
                    [[0.15, 0.70], [0.80, 0.35]],
                ]
            )
        ]
        actions = jnp.array([0, 1])
        qA = [jnp.array([[4.0, 2.0], [1.5, 5.0]])]
        pA = [jnp.array([[2.0, 1.0], [1.0, 2.5]])]
        qB = [jnp.array([[[5.0, 2.0], [2.0, 4.0]], [[1.5, 3.0], [4.5, 2.0]]])]
        pB = [jnp.array([[[2.5, 1.5], [1.5, 3.0]], [[1.0, 2.0], [2.5, 1.0]]])]

        vfe_t, vfe, info = maths.calc_vfe(
            qs,
            prior,
            obs=obs,
            A=A,
            B=B,
            past_actions=actions,
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            qA=qA,
            pA=pA,
            qB=qB,
            pB=pB,
            return_decomposition=True,
        )

        qs_np = np.asarray(qs[0])
        prior_np = np.asarray(prior[0])
        obs_np = np.asarray(obs[0])
        A_np = np.asarray(A[0])
        B_np = np.asarray(B[0])

        expected_vfe_t = []
        for t in range(qs_np.shape[0]):
            neg_entropy = float(np.sum(qs_np[t] * np.log(qs_np[t])))
            if t == 0:
                dynamics = float(-np.sum(qs_np[t] * np.log(prior_np)))
            else:
                b_t = B_np[..., int(actions[t - 1])]
                dynamics = float(
                    -np.sum(
                        qs_np[t][:, None]
                        * qs_np[t - 1][None, :]
                        * np.log(b_t)
                    )
                )
            accuracy = float(
                np.sum(
                    qs_np[t]
                    * np.sum(obs_np[t][:, None] * np.log(A_np), axis=0)
                )
            )
            expected_vfe_t.append(neg_entropy + dynamics - accuracy)

        expected_vfe_t = np.asarray(expected_vfe_t)
        expected_kl_A = _manual_dirichlet_kl(np.asarray(qA[0]), np.asarray(pA[0]))
        expected_kl_B = _manual_dirichlet_kl(np.asarray(qB[0]), np.asarray(pB[0]))

        np.testing.assert_allclose(np.asarray(vfe_t), expected_vfe_t, atol=1e-6)
        np.testing.assert_allclose(np.asarray(info["state_vfe"]), expected_vfe_t.sum(), atol=1e-6)
        np.testing.assert_allclose(np.asarray(info["parameter_kl_A"]), expected_kl_A, atol=1e-6)
        np.testing.assert_allclose(np.asarray(info["parameter_kl_B"]), expected_kl_B, atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(vfe),
            expected_vfe_t.sum() + expected_kl_A + expected_kl_B,
            atol=1e-6,
        )

    def test_calc_vfe_gradients_are_finite(self):
        qs = [jnp.array([0.60, 0.40])]
        prior = [jnp.array([0.55, 0.45])]
        obs = [jnp.array([1.0, 0.0])]

        def scalar_vfe(logits: jnp.ndarray) -> jnp.ndarray:
            A = [nn.softmax(logits, axis=0)]
            return maths.calc_vfe(
                qs,
                prior,
                obs=obs,
                A=A,
                A_dependencies=[[0]],
            )[1]

        gradients = grad(scalar_vfe)(jnp.array([[0.5, -0.1], [0.2, 0.4]]))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradients))))

    def test_update_posterior_states_return_info_includes_vfe(self):
        A = [jnp.array([[0.9, 0.2], [0.1, 0.8]])]
        prior = [jnp.array([0.55, 0.45])]
        obs = [jnp.array([[0.0, 1.0]])]

        qs_hist, info = inference.update_posterior_states(
            A=A,
            B=None,
            obs=obs,
            past_actions=None,
            prior=prior,
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            method="fpi",
            distr_obs=True,
            return_info=True,
        )

        expected_vfe_t, expected_vfe = maths.calc_vfe(
            [qs_hist[0][0]],
            prior,
            obs=[obs[0][0]],
            A=A,
            A_dependencies=[[0]],
        )

        self.assertEqual(qs_hist[0].shape, (1, 2))
        self.assertIn("vfe_t", info)
        self.assertIn("vfe", info)
        self.assertIn("vfe_components", info)
        np.testing.assert_allclose(np.asarray(info["vfe_t"]), np.asarray(expected_vfe_t), atol=1e-6)
        np.testing.assert_allclose(np.asarray(info["vfe"]), np.asarray(expected_vfe), atol=1e-6)

    def test_agent_infer_states_return_info_batches_sequence_vfe(self):
        A = [jnp.array([[0.90, 0.15], [0.10, 0.85]])]
        B = [
            jnp.array(
                [
                    [[0.85, 0.30], [0.20, 0.65]],
                    [[0.15, 0.70], [0.80, 0.35]],
                ]
            )
        ]
        D = [jnp.array([0.60, 0.40])]
        observations = [jnp.array([[0, 1, 1]])]
        actions = jnp.array([[[0], [1]]])

        agent = Agent(
            A=A,
            B=B,
            D=D,
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            num_controls=[2],
            inference_algo="mmp",
            inference_horizon=3,
            batch_size=1,
        )

        qs_hist, info = agent.infer_states(
            observations=observations,
            empirical_prior=agent.D,
            past_actions=actions,
            return_info=True,
        )

        expected_vfe_t, expected_vfe = maths.calc_vfe(
            [qs_hist[0][0]],
            [agent.D[0][0]],
            obs=[nn.one_hot(observations[0][0], 2)],
            A=[agent.A[0][0]],
            B=[agent.B[0][0]],
            past_actions=actions[0, :, 0],
            A_dependencies=[[0]],
            B_dependencies=[[0]],
        )

        self.assertEqual(qs_hist[0].shape, (1, 3, 2))
        self.assertEqual(info["vfe_t"].shape, (1, 3))
        self.assertEqual(info["vfe"].shape, (1,))
        np.testing.assert_allclose(
            np.asarray(info["vfe_t"][0]),
            np.asarray(expected_vfe_t),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.asarray(info["vfe"][0]),
            np.asarray(expected_vfe),
            atol=1e-6,
        )

    def test_sequence_return_info_accepts_missing_past_actions(self):
        A = [jnp.array([[0.90, 0.15], [0.10, 0.85]])]
        B = [
            jnp.array(
                [
                    [[0.85, 0.30], [0.20, 0.65]],
                    [[0.15, 0.70], [0.80, 0.35]],
                ]
            )
        ]
        prior = [jnp.array([0.60, 0.40])]
        obs = [jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])]

        qs_hist, info = inference.update_posterior_states(
            A=A,
            B=B,
            obs=obs,
            past_actions=None,
            prior=prior,
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            method="mmp",
            distr_obs=True,
            return_info=True,
        )

        self.assertEqual(qs_hist[0].shape, (3, 2))
        self.assertEqual(info["vfe_t"].shape, (3,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(info["vfe_t"]))))
        self.assertTrue(bool(jnp.isfinite(info["vfe"])))

    def test_calc_vfe_with_smoothed_exact_posterior_matches_bruteforce_sequence_vfe(self):
        prior = [jnp.array([0.55, 0.45])]
        A = [jnp.array([[0.92, 0.12], [0.08, 0.88]])]
        B = [
            jnp.array(
                [
                    [[0.85, 0.30], [0.20, 0.65]],
                    [[0.15, 0.70], [0.80, 0.35]],
                ]
            )
        ]
        obs_idx = jnp.array([0, 1, 1])
        obs = [nn.one_hot(obs_idx, 2)]
        actions = jnp.array([0, 1])

        filtered_qs = _filtered_history_single_factor("exact", A, B, prior, obs, actions)
        smoothed_qs, joint_qs = inference.smoothing_exact(filtered_qs, B, actions)
        _, vfe = maths.calc_vfe(
            smoothed_qs,
            prior,
            obs=obs,
            A=A,
            B=B,
            past_actions=actions,
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            joint_qs=joint_qs,
        )

        expected_vfe = _bruteforce_sequence_vfe(
            np.asarray(prior[0]),
            np.asarray(A[0]),
            np.asarray(B[0]),
            np.asarray(obs_idx),
            np.asarray(actions),
        )

        np.testing.assert_allclose(np.asarray(vfe), expected_vfe, atol=1e-6)

    def test_calc_vfe_with_smoothed_ovf_posterior_matches_bruteforce_sequence_vfe(self):
        prior = [jnp.array([0.55, 0.45])]
        A = [jnp.array([[0.92, 0.12], [0.08, 0.88]])]
        B = [
            jnp.array(
                [
                    [[0.85, 0.30], [0.20, 0.65]],
                    [[0.15, 0.70], [0.80, 0.35]],
                ]
            )
        ]
        obs_idx = jnp.array([0, 1, 1])
        obs = [nn.one_hot(obs_idx, 2)]
        actions = jnp.array([0, 1])

        filtered_qs = _filtered_history_single_factor("ovf", A, B, prior, obs, actions)
        smoothed_qs, joint_qs = inference.smoothing_ovf(filtered_qs, B, actions)
        _, vfe = maths.calc_vfe(
            smoothed_qs,
            prior,
            obs=obs,
            A=A,
            B=B,
            past_actions=actions,
            A_dependencies=[[0]],
            B_dependencies=[[0]],
            joint_qs=joint_qs,
        )

        expected_vfe = _bruteforce_sequence_vfe(
            np.asarray(prior[0]),
            np.asarray(A[0]),
            np.asarray(B[0]),
            np.asarray(obs_idx),
            np.asarray(actions),
        )

        np.testing.assert_allclose(np.asarray(vfe), expected_vfe, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
