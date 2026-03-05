# Release Notes

## 1.0.0 (in progress)

- Breaking (JAX API): `rollout()` extras now use `info["neg_efe"]` instead of
  `info["G"]`.
- JAX examples and docs now use `q_pi, neg_efe = agent.infer_policies(qs)` for
  policy-score outputs.
- SPM notation mapping is now explicit in docs: SPM's `G` corresponds to
  `neg_efe = -EFE`.
