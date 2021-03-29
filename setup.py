import setuptools

setuptools.setup(
    name="pymdp",
    version="0.0.1",
    description=("A Python-based implementation of active inference for Markov Decision Processes"),
    license="Apache 2.0",
    url="https://github.com/infer-actively/pymdp",
    packages=[
        "pymdp",
        "pymdp.core",
        "pymdp.distributions",
        "pymdp.agent",
        "pymdp.envs",
    ],
)
