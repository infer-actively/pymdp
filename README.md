
<p align='center'>
  <a href='https://github.com/alec-tschantz/pymdp'>
    <img src='.github/logo.png' />
  </a> 
</p>

An Python-based implementation of `active inference` for Markov Decision Processes,
based on functions from the `SPM` [implementation](https://www.fil.ion.ucl.ac.uk/spm/doc/)

## Status

![status](https://img.shields.io/badge/status-development-orange)

This package is under active development. If you would like to contribute, please refer to [this file](CONTRIBUTING.md)

## Installation and Usage

In order to use this code, download the `pymdp` folder into your project
folder. You can now use all classes and functions via `import pymdp`

## Getting started

For an illustrative tutorial for how to use the functionalities of the `Agent()` class, which is used to perform active inference using the core functionality of `pymdp`, we recommend
going through the Jupyter notebooks in the `pymdp/examples/` folder. `tmaze_demo.ipynb` is a good place to start, and provides a step-by-step walkthrough of how to build an instance
of `Agent()`, sample observations from the generative process, and perform active inference. 

THe `agent_demo.ipynb` notebook also provides a more 'naked' implementation of how to build and encode a generative model in terms of straight numpy arrays. Unlike the `tmaze_demo`, this doesn't rely on an environment constructor class, whose methods can build the `A`, `B`, etc. matrices for you.

## Requirements 

This code is written in [Python 3.x](https://www.python.org) and uses 
the following packages:

* [NumPy](https://github.com/numpy/numpy)
* [SciPy](http://numpy.scipy.org/)
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [Seaborn](https://seaborn.pydata.org/)

You can install the relevant package versions with conda (YMMV):
```bash
conda env create -f environment.yml
```

If you would like to contribute to this repo, you may have more success with venv and pip
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