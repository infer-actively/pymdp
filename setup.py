import setuptools

setuptools.setup(
    name="inferactively",
    version="0.0.1",
    description=("An Python-based implementation of active inference for Markov Decision Processes"),
    license="Apache 2.0",
    url="https://github.com/alec-tschantz/infer-actively",
    packages=[
        "inferactively",
        "inferactively.core",
        "inferactively.distributions",
        "inferactively.agent",
        "inferactively.envs",
    ],
)
