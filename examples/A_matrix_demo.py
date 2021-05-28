# %%

import os
import sys
import pathlib

import numpy as np
import itertools
import pandas as pd

path = pathlib.Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

import pymdp.core.utils as utils
from pymdp.core.utils import create_A_matrix_stub
from pymdp.core.algos import run_fpi

# %% Create an empty A matrix
model_labels = {
            "observations": {
                "grass_observation": [
                    "wet",
                    "dry"            
                    ],
                "weather_observation": [
                    "clear",
                    "rainy",
                    "cloudy"
                ]
            },
            "states": {
                "did_it_rain": ["rained", "did_not_rain"],
                "was_sprinkler_on": ["on", "off"],
            },
        }

A_stub = create_A_matrix_stub(model_labels)

# %% Option 1: fill out A matrix 'offline' (e.g. in an excel spreadsheet)

excel_dir = 'tmp_dir'
if not os.path.exists(excel_dir):
    os.mkdir(excel_dir)

excel_path = os.path.join(excel_dir, 'my_a_matrix.xlsx')

A_stub.to_excel(excel_path)

# now go fill out the A matrix in the excel file you've written to disk
# %% Now read it back in
A_stub = read_A_matrix(excel_path)

# %% Option 2: fill out the A matrix here in Python, using our knowledge of the dependencies in the system

# A_stub.loc[('grass_observation','wet'),('rained', 'on')] = 1.0

# A_stub.loc[('grass_observation','wet'),('rained', 'off')] = 0.7
# A_stub.loc[('grass_observation','dry'),('rained', 'off')] = 0.3

# A_stub.loc[('grass_observation','wet'),('did_not_rain', 'on')] = 0.5
# A_stub.loc[('grass_observation','dry'),('did_not_rain', 'on')] = 0.5

# A_stub.loc[('grass_observation','dry'),('did_not_rain', 'off')] = 1.0

# A_stub.loc[('grass_observation','wet'),('rained', 'on')] = 1.0


# A_stub.loc[('weather_observation','clear'),('rained')] = 0.2
# A_stub.loc[('weather_observation','rainy'),('rained')] = 0.2
# A_stub.loc[('weather_observation','cloudy'),('rained')] = 0.2

# A_stub.loc['weather_observation','rained'] = np.tile(np.array([0.1, 0.65, 0.25]).reshape(-1,1), (1,2)) 

# A_stub.loc[('weather_observation'),('did_not_rain')] = np.tile(np.array([0.9, 0.05, 0.05]).reshape(-1,1), (1,2)) 

# %% now convert the A matrix into a sequence of appopriately shaped numpy arrays

A = utils.convert_A_stub_to_ndarray(A_stub, model_labels)

num_obs, _, n_states, _ = utils.get_model_dimensions_from_labels(model_labels)

obs_idx = [np.random.randint(o_dim) for o_dim in num_obs]

observation = utils.obj_array_zeros(num_obs)

for g, modality_name in enumerate(model_labels['observations'].keys()):
    observation[g][obs_idx[g]] = 1.0
    print('%s: %s'%(modality_name, model_labels['observations'][modality_name][obs_idx[g]]))

qs = run_fpi(A, observation, num_obs, n_states, prior=None, num_iter=10, dF=1.0, dF_tol=0.001)

print('Belief that it rained: %.2f'%(qs[0][0]))
print('Belief that the sprinkler was on: %.2f'%(qs[1][0]))


# %%
