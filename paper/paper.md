---
title: 'pymdp: A Python library for active inference in discrete state spaces'
tags:
  - Python
  - active inference
  - Markov Decision Process
  - POMDP
  - MDP
  - Reinforcement Learning
  - Artificial Intelligence
  - Bayesian inference
  - free energy principle
authors:
  - name: Conor Heins^[corresponding author]
    affiliation: "1, 2, 3, 4"
  - name: Beren Millidge
    affiliation: "4, 5"
  - name: Daphne Demekas
    affiliation: "6"
  - name: Brennan Klein
    orcid: 0000-0001-8326-5044
    affiliation: "4, 7, 8"
  - name: Karl Friston
    affiliation: "9" 
  - name: Iain D. Couzin
    affiliation: "1, 2, 3" 
  - name: Alexander Tschantz^[corresponding author]
    affiliation: "4, 10, 11"
affiliations:
 - name: Department of Collective Behaviour, Max Planck Institute of Animal Behavior, 78457 Konstanz, Germany
   index: 1
 - name: Centre for the Advanced Study of Collective Behaviour, 78457 Konstanz, Germany
   index: 2
 - name: Department of Biology, University of Konstanz, 78457 Konstanz, Germany
   index: 3
 - name: VERSES Research Lab, Los Angeles, California, USA
   index: 4
 - name: MRC Brain Networks Dynamics Unit, University of Oxford, Oxford, UK
   index: 5
 - name: Department of Computing, Imperial College London, London, UK
   index: 6
 - name: Network Science Institute, Northeastern University, Boston, MA, USA
   index: 7
 - name: Laboratory for the Modeling of Biological and Socio-Technical Systems, Northeastern University, Boston, USA
   index: 8
 - name: Wellcome Centre for Human Neuroimaging, Queen Square Institute of Neurology, University College London, London WC1N 3AR, UK
   index: 9
 - name: Sussex AI Group, Department of Informatics, University of Sussex, Brighton, UK
   index: 10
 - name: Sackler Centre for Consciousness Science, University of Sussex, Brighton, UK
   index: 11
  
date: 14 January 2022
bibliography: paper.bib
---

# Statement of Need

Active inference is an account of cognition and behavior in complex systems which brings together action, perception, and learning under the theoretical mantle of Bayesian inference [@friston_reinforcement_2009; @friston_active_2012; @friston2015active; @friston2017process]. Active inference has seen growing applications in academic research, especially in fields that seek to model human or animal behavior [@parr2020prefrontal; @holmes2021active; @adams2021everything]. The majority of applications have focused on cognitive neuroscience, with a particular focus on modelling decision-making under uncertainty [@schwartenbeck2015dopaminergic; @smith2020imprecise; @smith2021greater]. Nonetheless, the framework has broad applicability and has recently been applied to diverse disciplines, ranging from computational models of psychopathology [@montague2012computational; @smith2021greater], control theory [@baltieri2019pid; @millidge2020relationship; @baioumy2021towards] and reinforcement learning  [@tschantz2020reinforcement; @tschantz2020scaling; @sajid2021active; @fountas2020deep; @millidge2020deep], through to social cognition [@adams2021everything; @wirkuttis2021leading; @tison2021communication] and even real-world engineering problems [@martinez2021probabilistic; @moreno2021pid; @fox2021active]. While in recent years, some of the code arising from the active inference literature has been written in open source languages like Python and Julia  [@ueltzhoffer2018deep; @van2019simulating; @tschantz2020learning; @ccatal2020learning; @millidge2020deep], to-date, the most popular software for simulating active inference agents is the `DEM` toolbox of `SPM` [@friston2008variational; @smith2022step], a MATLAB library originally developed for the statistical analysis and modelling of neuroimaging data [@penny2011statistical]. `DEM` contains a reliable, reproducible set of functions for studying active inference, but the use of the toolbox can be restrictive for researchers in settings where purchasing a MATLAB license is financially costly. And although active inference researchers have relied heavily on `DEM` for simulating and fitting models of behavior, most of its functionality is restricted to single MATLAB scripts or functions, particularly one called `spm_MDP_VB_X.m`, that lack modularity and often must be customized for applications on a domain-specific basis. Increasing interest in active inference, manifested both in terms of sheer number of cited research papers as well as diversifying applications across disciplines, has thus created a need for generic, widely-available, and user-friendly code for simulating active inference in open-source scientific computing languages like Python. The software we present here, [`pymdp`](https://github.com/infer-actively/pymdp), represents a significant step in this direction: namely, we provide the first open-source package for simulating active inference with discrete state-space generative models. The name `pymdp` derives from the fact that the package is written in the **Py**thon programming language and concerns discrete, Markovian generative models of decision-making, which take the form of Markov Decision Processes or **MDP**s.

`pymdp` is a Python package that is directly inspired by the active inference routines contained in `DEM`. However, `pymdp`  is has a modular, flexible structure that allows researchers to build and simulate active inference agents quickly and with a high degree of customization. We developed `pymdp` in the hopes that it will increase the accessibility and exposure of the active inference framework to researchers, engineers, and developers with diverse disciplinary backgrounds. In the spirit of open-source software, we also hope that it spurs new innovation, development, and collaboration in the growing active inference and wider Bayesian modelling communities. For additional pedagogical and technical resources on `pymdp`, we refer the reader to the package's github repository. We also encourage more technically-interested readers to consult a companion preprint article that includes technical material covering the mathematics of active inference in discrete state spaces [@heins2022pymdp_arxiv].

# Summary

`pymdp` offers a suite of robust, tested, and modular routines for simulating active inference agents equipped with *partially-observable Markov Decision Process* (POMDP) generative models. Mathematically,  a POMDP comprises a joint distribution over observations $o$, hidden states $s$, control states $u$ and hyperparameters $\phi$: $P(o, s, u, \phi)$. This joint distribution further factorizes into a set of categorical and Dirichlet distributions: the likelihoods and priors of the generative model. With `pymdp`, one can build a generative model using a set of prior and likelihood distributions, initialize an agent, and then link it to an external environment to run active inference processes - all in a few lines of code. The `Agent` and `Env` (environment) APIs of `pymdp` are built according to the standardized framework of OpenAIGym commonly used in reinforcement learning, where an agent and environment object recursively exchange observations and actions over time [@brockman2016openai].

# Introduction

Simulations of active inference are commonly performed in discrete time and space [@friston2015active; @da2020active]. This is partially motivated by the mathematical tractability of performing inference with discrete probability distributions, but also by the intuition of modelling choice behavior as a sequence of discrete, mutually-exclusive choices in, e.g., psychophysics or decision-making experiments. The most popular generative models -- used to realize active inference in this context -- are partially-observable Markov Decision Processes or *POMDPs* [@kaelbling1998planning]. POMDPs are state-space models that model the environment in terms of hidden states that stochastically change over time, as a function of both the current state of the environment as well as the behavioral output of an agent (control states or actions). Crucially, the environment is *partially-observable*, i.e. the hidden states are not directly observed by the agent, but can only be inferred through observations that relate to hidden states in a probabilistic manner, such that observations are modelled as being generated stochastically from the current hidden state. This necessitates both "perceptual" inference of hidden states as well as control.

As such, in most POMDP problems, an agent is tasked with inferring the hidden state of its environment and then choosing a sequence of control states or actions to change hidden states in a way that leads to desired outcomes (maximizing reward, or occupancy within some preferred set of states). 

# Usage 

In order to enhance the user-friendliness of `pymdp` without sacrificing flexibility, we have built the library to be highly modular and customizable, such that agents in `pymdp` can be specified at a variety of levels of abstraction with desired parameterisations. The methods of the `Agent` class can thus be called in any particular order, depending on the application, and furthermore they can be specified with various keyword arguments that entail choices of implementation details at lower levels.

By retaining a modular structure throughout the package's dependency hierarchy, `pymdp` also affords the ability to flexibly compose different low level functions. This allows users to customize and integrate their active inference loops with desired inference algorithms and policy selection routines. For instance, one could sub-class the `Agent` class and write a customized `step()` function, that combines whichever components of active inference one is interested in.   

# Related software packages

The `DEM` toolbox within `SPM` in MATLAB is the current gold-standard in active inference modelling. In particular, simulating an active inference process in `DEM` consists of defining the generative model in terms of a fixed set of matrices and vectors, and then calling the `spm_MDP_VB_X.m` function to simulate a sequence of trials. `pymdp`, by contrast, provides a user-friendly and modular development experience, with core functionality split up into different libraries that separately perform the computations of active inference in a standalone fashion. Moreover, `pymdp` provides the user the ability to write an active inference process at different levels of abstraction depending on the user's level of expertise or skill with the package -- ranging from the high level `Agent` functionality, which allows the user to define and simulate an active inference agent in just a few lines of code, all the way to specifying a particular variational inference algorithm (e.g., marginal-message passing [@parr2019neuronal]) for the agent to use during state estimation. In the `DEM` toolbox of `SPM`, this would require setting undocumented flags or else manually editing the routines in `spm_MDP_VB_X.m` to enable or disable bespoke functionality. There has been one recent attempt at creating a comprehensive user-guide for building active inference agents in `DEM` [@smith2022step], though to our knowledge there has not been a package devoted to the open source development of these powerful software tools.

A recent related, but largely non-overlapping project is [ForneyLab](https://github.com/biaslab/ForneyLab.jl), which provides a set of Julia libraries for performing approximate Bayesian inference via message passing on Forney Factor Graphs [@ForneyLab2019]. Notably, this package has also seen several applications in simulating active inference processes, using ForneyLab as the backend for the inference algorithms employed by an active inference agent [@van2019simulating; @vanderbroeck2019active; @ergul2020learning; @10.1162/neco_a_01427]. While ForneyLab focuses on including a rigorous set of message passing routines that can be used to simulate active inference agents, `pymdp` is specifically designed to help users quickly build agents (regardless of their underlying inference routines) and plug them into arbitrary environments to run active inference in a few easy steps.

# Funding Statement
CH and IDC acknowledge support from the Office of Naval Research grant (ONR, N00014- 64019-1-2556), with IDC further acknowledging support from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement (ID: 860949), the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy-EXC 2117- 422037984, and the Max Planck Society. KF is supported by funding for the Wellcome Centre for Human Neuroimaging (Ref: 205103/Z/16/Z) and the Canada-UK Artificial Intelligence Initiative (Ref: ES/T01279X/1). CH, DD, and BK acknowledge the support of a grant from the John Templeton Foundation (61780). The opinions expressed in this publication are those of the author(s) and do not necessarily reflect the views of the John Templeton Foundation.

# Acknowledgements

The authors would like to thank Dimitrije Markovic, Arun Niranjan, Sivan Altinakar, Mahault Albarracin, Alex Kiefer, Magnus Koudahl, Ryan Smith, Casper Hesp, and Maxwell Ramstead for discussions and feedback that contributed to development of `pymdp`. We would also like to thank Thomas Parr for pointing out a technical error in an earlier version of the arXiv preprint for this work. Finally, we are grateful to the many users of `pymdp` whose feedback and usage of the package have contributed to its continued improvement and development.

# References