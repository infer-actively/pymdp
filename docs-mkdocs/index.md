# Welcome to pymdp's documentation!

`pymdp` is a Python package for simulating Active Inference agents in
discrete-state Markov Decision Process environments. For background and
motivation, see the companion JOSS paper:
[`pymdp: A Python library for active inference in discrete state spaces`](https://joss.theoj.org/papers/10.21105/joss.04098).

This documentation focuses on the JAX backend (the default backend since the 1.0.0 release), which supports:

<ul>
  <li>accelerator-friendly execution (CPU/GPU/TPU) and JIT compilation</li>
  <li>differentiable workflows via autodiff</li>
  <li>scalable batched simulations and model-fitting workflows (for example, with NumPyro or <a href="https://github.com/dimarkov/pybefit">pybefit</a>)</li>
</ul>

![Cue-chaining demo](assets/cue-chaining-demo.gif){ width="880" }

## Start here
- [Installation](getting-started/installation.md)
- [Quickstart (JAX)](getting-started/quickstart-jax.md)
- [NumPy/legacy to JAX migration guide](migration/numpy-to-jax.md)
- [Using rollout() for compiled active inference loops](guides/rollout-active-inference-loop.md)
- [PymdpEnv and custom environments](guides/pymdp-env.md)
- [Thinking in generative models in `pymdp`](guides/generative-model-structure.md)
- [Notebook gallery](tutorials/notebooks/index.md)

## API reference
- [API overview](api/index.md)

## Legacy docs
Legacy/NumPy material is preserved under an archive path:

- [Legacy/NumPy archive](legacy/index.md)
