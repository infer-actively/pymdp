Env
========

The OpenAIGym-inspired ``Env`` base class is the main API that represents the environmental dynamics or "generative process" with
which agents exchange observations and actions

Base class
----------
.. autoclass:: pymdp.envs.Env

Specific environment implementations
----------

All of the following dynamics inherit from ``Env`` and have the
same general usage as above.

.. autosummary::
   :nosignatures:

    pymdp.envs.GridWorldEnv
    pymdp.envs.DGridWorldEnv
    pymdp.envs.VisualForagingEnv
    pymdp.envs.TMazeEnv
    pymdp.envs.TMazeEnvNullOutcome
