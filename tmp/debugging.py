import os
import sys
import unittest

import numpy as np
from scipy.io import loadmat

sys.path.append(".")
from inferactively import Categorical, Dirichlet  # nopep8

c = Categorical(dims=[3, 4])
print(c.cross())