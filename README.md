
<p align='center'>
  <a href='https://github.com/infer-actively/pymdp'>
    <img src='.github/pymdp_logo_2-removebg.png' />
  </a> 
</p>

A Python package for simulating Active Inference agents in Markov Decision Process environments.
Please see our companion paper, published in the Journal of Open Source Software: ["pymdp: A Python library for active inference in discrete state spaces"](https://joss.theoj.org/papers/10.21105/joss.04098) for an overview of the package and its motivation. For a more in-depth, tutorial-style introduction to the package and a mathematical overview of active inference in Markov Decision Processes, see the [longer arxiv version](https://arxiv.org/abs/2201.03904) of the paper.

This package is hosted on the [`infer-actively`](https://github.com/infer-actively) GitHub organization, which was built with the intention of hosting open-source active inference and free-energy-principle related software.

Most of the low-level mathematical operations are [NumPy](https://github.com/numpy/numpy) ports of their equivalent functions from the `SPM` [implementation](https://www.fil.ion.ucl.ac.uk/spm/doc/) in MATLAB. We have benchmarked and validated most of these functions against their SPM counterparts.

## Status

![status](https://img.shields.io/badge/status-active-green)
![PyPI version](https://img.shields.io/pypi/v/inferactively-pymdp)
[![Documentation Status](https://readthedocs.org/projects/pymdp-rtd/badge/?version=docs-rehaul)](https://pymdp-rtd.readthedocs.io/en/docs-rehaul/?badge=docs-rehaul)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04098/status.svg)](https://doi.org/10.21105/joss.04098)


# ``pymdp`` in action

Here's a visualization of ``pymdp`` agents in action. One of the defining features of active inference agents is the drive to maximize "epistemic value" (i.e. curiosity). Equipped with such a drive in environments with uncertain yet disclosable hidden structure, active inference can ultimately allow agents to simultaneously learn about the environment as well as maximize reward.

The simulation below (see associated notebook [here](https://pymdp-rtd.readthedocs.io/en/latest/notebooks/cue_chaining_demo.html)) demonstrates what might be called "epistemic chaining," where an agent (here, analogized to a mouse seeking food) forages for a chain of cues, each of which discloses the location of the subsequent cue in the chain. The final cue (here, "Cue 2") reveals the location a hidden reward. This is similar in spirit to "behavior chaining" used in operant conditioning, except that here, each successive action in the behavioral sequence doesn't need to be learned through instrumental conditioning. Rather, active inference agents will naturally forage the sequence of cues based on an intrinsic desire to disclose information. This ultimately leads the agent to the hidden reward source in the fewest number of moves as possible.

You can run the code behind simulating tasks like this one and others in the **Examples** section of the [official documentation](https://pymdp-rtd.readthedocs.io/en/latest/).

<!-- 
<p align="center">
  <img src=".github/chained_cue_navigation_v1.gif" width="50%" height="50%"/>
  <img src=".github/chained_cue_navigation_v2.gif" width="50%" height="50%"/>
</p> -->

<!-- ![alt](.github/chained_cue_navigation_v1.gif) | ![alt](.github/chained_cue_navigation_v2.gif) -->

<table><tr>
<td> 
  <p align="center" style="padding: 10px">
    <img src=".github/chained_cue_navigation_v1.gif" width="100%" height="50%"/>
    <br>
    <em style="color: grey">Cue 2 in Location 1, Reward on Top</em>
  </p> 
</td>
<td> 
  <p align="center">
    <img src=".github/chained_cue_navigation_v2.gif" width="100%" height="50%"/>
    <br>
    <em style="color: grey">Cue 2 in Location 3, Reward on Bottom</em>
  </p> 
</td>
</tr></table>

## Quick-start: Installation and Usage

In order to use `pymdp` to build and develop active inference agents, we recommend installing it with the the package installer [`pip`](https://pip.pypa.io/en/stable/), which will install `pymdp` locally as well as its dependencies. This can also be done in a virtual environment (e.g. with `venv`). 

When pip installing `pymdp`, use the package name `inferactively-pymdp`:

```bash
pip install inferactively-pymdp
```

Once in Python, you can then directly import `pymdp`, its sub-packages, and functions.

```python
from jax import random as jr
from pymdp import utils
from pymdp.agent import Agent

key = jr.PRNGKey(0)
keys = jr.split(key, 3)

num_obs = [3, 5]
num_states = [3, 2]
num_controls = [3, 1]

A = utils.random_A_array(keys[0], num_obs, num_states)
B = utils.random_B_array(keys[1], num_states, num_controls)
C = utils.list_array_uniform([[no] for no in num_obs])

agent = Agent(A=A, B=B, C=C)
observation = [1, 4]

qs = agent.infer_states(observation, empirical_prior=agent.D)
q_pi, neg_efe = agent.infer_policies(qs)

action_keys = jr.split(keys[2], agent.batch_size + 1)
action = agent.sample_action(q_pi, rng_key=action_keys[1:])
```

## Getting started / introductory material

We recommend starting with the JAX-first [official documentation](https://pymdp-rtd.readthedocs.io/en/latest/) for the repository, which provides practical guides, curated notebooks, and generated API references.

For new users to `pymdp`, we specifically recommend stepping through following three Jupyter notebooks (can also be used on Google Colab):

- [Quickstart (JAX)](https://pymdp-rtd.readthedocs.io/en/latest/getting-started/quickstart-jax/)
- [NumPy/legacy to JAX migration guide](https://pymdp-rtd.readthedocs.io/en/latest/migration/numpy-to-jax/)
- [`rollout()` active inference loop guide](https://pymdp-rtd.readthedocs.io/en/latest/guides/rollout-active-inference-loop/)

We also have (and are continuing to build) a series of notebooks that walk through active inference agents performing different types of tasks in the [Notebook Gallery](https://pymdp-rtd.readthedocs.io/en/latest/tutorials/notebooks/).

## Contributing

This package is under active development. If you would like to contribute, please refer to [this file](CONTRIBUTING.md)

If you would like to contribute to this repo, we recommend using venv and pip
```bash
cd <path_to_repo_fork>
python3 -m venv env
source env/bin/activate
pip install -e .  # This will install pymdp as a local dev package
```

You should then be able to run tests locally with `pytest`
```bash
pytest test
```

## Citing `pymdp`
If you use `pymdp` in your work or research, please consider citing our [paper](https://joss.theoj.org/papers/10.21105/joss.04098) (open-access) published in the Journal of Open-Source Software:

```
@article{Heins2022,
  doi = {10.21105/joss.04098},
  url = {https://doi.org/10.21105/joss.04098},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {73},
  pages = {4098},
  author = {Conor Heins and Beren Millidge and Daphne Demekas and Brennan Klein and Karl Friston and Iain D. Couzin and Alexander Tschantz},
  title = {pymdp: A Python library for active inference in discrete state spaces},
  journal = {Journal of Open Source Software}
}
```

For a more in-depth, tutorial-style introduction to the package and a mathematical overview of active inference in Markov Decision Processes, you can also consult the [longer arxiv version](https://arxiv.org/abs/2201.03904) of the paper.

## Authors

- Conor Heins [@conorheins](https://github.com/conorheins)
- Alec Tschantz [@alec-tschantz](https://github.com/alec-tschantz)
- Beren Millidge [@BerenMillidge](https://github.com/BerenMillidge)
- Brennan Klein [@jkbren](https://github.com/jkbren)
- Arun Niranjan [@Arun-Niranjan](https://github.com/Arun-Niranjan)
- Daphne Demekas [@daphnedemekas](https://github.com/daphnedemekas)
- Aswin Paul [@aswinpaul](https://github.com/aswinpaul)
- Tim Verbelen [@tverbele](https://github.com/tverbele)
- Dimitrije Markovic [@dimarkov](https://github.com/dimarkov)
