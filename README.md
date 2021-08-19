
<p align='center'>
  <a href='https://github.com/infer-actively/pymdp'>
    <img src='.github/logo.png' />
  </a> 
</p>

A Python package for simulating Active Inference agents in Markov Decision Process environments. 
This package is hosted on the [`infer-actively`](https://github.com/infer-actively) GitHub organization, which was built with the intention of hosting open-source active inference and free-energy-principle related software.

Most of the low-level mathematical operations are [NumPy](https://github.com/numpy/numpy) ports of their equivalent functions from the `SPM` [implementation](https://www.fil.ion.ucl.ac.uk/spm/doc/) in MATLAB. We have benchmarked and validated most of these functions against their SPM counterparts.

## Status

![status](https://img.shields.io/badge/status-development-orange)

## Installation and Usage

In order to use `pymdp` to build and develop active inference agents, we recommend installing it with the the package installer [`pip`](https://pip.pypa.io/en/stable/), which will install `pymdp` locally as well as its dependencies. This can also be done in a virtual environment (e.g. with `venv`). 

When pip installing `pymdp`, use the package name `inferactively-pymdp`:

```bash
pip install inferactively-pymdp
```

Once in Python, you can then directly import `pymdp`, its sub-packages, and functions.

```bash

import pymdp
from pymdp import utils
from pymdp.agent import Agent

num_obs = [3, 5] # observation modality dimensions
num_states = [3, 2, 2] # hidden state factor dimensions
num_controls = [3, 1, 1] # control state factor dimensions
A_matrix = utils.random_A_matrix(num_obs, num_states) # create sensory likelihood (A matrix)
B_matrix = utils.random_B_matrix(num_states, num_controls) # create transition likelihood (B matrix)

C_vector = utils.obj_array_uniform(num_obs) # uniform preferences

# instantiate a quick agent using your A, B and C arrays
my_agent = Agent( A = A_matrix, B = B_matrix, C = C_vector)

# give the agent a random observation and get the optimized posterior beliefs

observation = [1, 4] # a list specifying the indices of the observation, for each observation modality

qs = my_agent.infer_states(observation) # get posterior over hidden states (a multi-factor belief)

# Do active inference

q_pi, neg_efe = my_agent.infer_policies() # return the policy posterior and return (negative) expected free energies of each policy as well

action = my_agent.sample_action() # sample an action

# ... and so on ...
```


## Getting started / pedagogical materials

The highest level API that `pymdp` currently offers is the `Agent()` class - this is a class whose methods abstract the core mathematical operations involved in active inference, which are lower level libraries within `pymdp` (e.g. `pymdp.inference`). 

For an illustrative tutorial on how to instantiate and use an `Agent()`, we recommend going through the Jupyter notebooks in the `pymdp/examples/` folder - the `gridworld_tutorial_1.ipynb` notebook and the `gridworld_tutorial_2.ipynb` notebooks are a good place to start. Special thanks to [Beren Millidge](https://github.com/BerenMillidge) and [Daphne Demekas](https://github.com/daphnedemekas) for their help constructing the gridworld tutorials, which were originally based on a set of tutorial notebooks written by [Alec Tschantz](https://github.com/alec-tschantz).

In order to go through these pedagogical materials (which are not included if you `pip install` the package), we recommend following these steps:

1. Clone (`git clone https://github.com/infer-actively/pymdp.git`) or download the repository locally and then `cd` into it.
2. Start a virtual environment (with either `venv` or `conda`) & install the requirements.
   1. If you're using `conda`:
      ```bash
      conda env create -f environment.yml
      ```
   2. If you're using `venv`
      ```bash
      cd <path_to_repo_fork>
      python3 -m venv env
      source env/bin/activate
      pip install -r requirements.txt
      ```
3. then run the IPython notebooks interactively, either as a straight Jupyter Notebook or on a platform that supports IPython notebooks (e.g. VSCode's IPython extension, Jupyterlab, etc.).

## Contributing

This package is under active development. If you would like to contribute, please refer to [this file](CONTRIBUTING.md)

If you would like to contribute to this repo, we recommend using venv and pip
```bash
cd <path_to_repo_fork>
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e ./ # This will install pymdp as a local dev package
```

You should then be able to run tests locally with `pytest`
```bash
pytest test
```

## Authors

- Conor Heins [@conorheins](https://github.com/conorheins)
- Alec Tschantz [@alec-tschantz](https://github.com/alec-tschantz)
- Beren Millidge [@BerenMillidge](https://github.com/BerenMillidge)
- Brennan Klein [@jkbren](https://github.com/jkbren)
- Arun Niranjan [@Arun-Niranjan](https://github.com/Arun-Niranjan)
- Daphne Demekas [@daphnedemekas](https://github.com/daphnedemekas)
