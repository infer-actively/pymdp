# %%
import os
import sys
import pathlib

import numpy as np
import itertools
import pandas as pd
from pandas import ExcelWriter

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

import pymdp.utils as utils
from pymdp.utils import create_B_matrix_stubs, get_model_dimensions_from_labels
# %% Create an empty B matrix

model_labels = {
            "observations": {
                "reward outcome": [
                    "win",
                    "loss"            
                    ]
            },
            "states": {
                "location": ["start", "arm1", "arm2"],
                "bandit_state": ["high_rew", "low_rew"]
            },
            "actions": {
                "arm_control": ["play_arm1", "play_arm2"],
                "bandit_state_control": ["null"]
            }
        }

B_stubs = create_B_matrix_stubs(model_labels)

# %% Option 1: Write the B matrices to a multi-sheet excel file
xls_dir = 'tmp_dir'
if not os.path.exists(xls_dir):
    os.mkdir(xls_dir)

xls_fpath = os.path.join(xls_dir, 'my_b_matrices.xlsx')

with ExcelWriter(xls_fpath) as writer:
    for factor_name, B_stub_f in B_stubs.items():
        B_stub_f.to_excel(writer,'%s' % factor_name)
# After filling them in, read them back in and convert
B_stubs_read_in = utils.read_B_matrices(xls_fpath)
B = utils.convert_B_stubs_to_ndarray(B_stubs_read_in, model_labels)

# %% Option 2: Code in the dependencies in pandas directly with multi-indexing operations

# B_stubs['location'].loc['arm1',('start', 'play_arm1')] = 1.0
# B_stubs['location'].loc['arm1',('arm1', 'play_arm1')] = 1.0
# B_stubs['location'].loc['arm1',('arm2', 'play_arm1')] = 1.0

# B_stubs['location'].loc['arm2',('start', 'play_arm2')] = 1.0
# B_stubs['location'].loc['arm2',('arm1', 'play_arm2')] = 1.0
# B_stubs['location'].loc['arm2',('arm2', 'play_arm2')] = 1.0

# B_stubs['bandit_state'].loc['high_rew', ('high_rew', 'null')] = 1.0
# B_stubs['bandit_state'].loc['low_rew', ('low_rew', 'null')] = 1.0
# B = utils.convert_B_stubs_to_ndarray(B_stubs, model_labels)



