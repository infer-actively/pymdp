import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="inferactively-pymdp",
    version="0.0.7.1",
    author="infer-actively",
    author_email="conor.heins@gmail.com",
    description= ("A Python package for solving Markov Decision Processes with Active Inference"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/infer-actively/pymdp",
    python_requires='>3.7',
    install_requires =[
    'attrs>=20.3.0', 
    'cycler>=0.10.0',
    'iniconfig>=1.1.1',
    'kiwisolver>=1.3.1', 
    'matplotlib>=3.1.3',
    'nose>=1.3.7', 
    'numpy>=1.19.5', 
    'openpyxl>=3.0.7',
    'packaging>=20.8', 
    'pandas>=1.2.4',
    'Pillow>=8.2.0',
    'pluggy>=0.13.1', 
    'py>=1.10.0',
    'pyparsing>=2.4.7',
    'pytest>=6.2.1',
    'python-dateutil>=2.8.1',
    'pytz>=2020.5',
    'scipy>=1.6.0',
    'seaborn>=0.11.1',
    'six>=1.15.0',
    'toml>=0.10.2',
    'typing-extensions>=3.7.4.3',
    'xlsxwriter>=1.4.3',
    'sphinx-rtd-theme>=0.4',
    'myst-nb>=0.13.1',
    'autograd>=1.3'
    ],
    packages=[
        "pymdp",
        "pymdp.envs",
        "pymdp.algos"
    ],
    include_package_data=True,
    keywords=[
        "artificial intelligence",
        "active inference",
        "free energy principle"
        "information theory",
        "decision-making",
        "MDP",
        "Markov Decision Process",
        "Bayesian inference",
        "variational inference",
        "reinforcement learning"
    ],
    classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
  ],
)

