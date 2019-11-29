# %%
import os

import numpy as np
from scipy.io import loadmat

from inferactively import Categorical

# %%
array_path = os.path.join(os.getcwd(), "tests/data/dot_a.mat")
mat_contents = loadmat(file_name=array_path)

A = mat_contents["A"]
obs = mat_contents["o"]
states = mat_contents["s"]
states = np.array(states, dtype=object)

result_1 = mat_contents["result1"]
result_2 = mat_contents["result2"]
result_3 = mat_contents["result3"]

A = Categorical(values=A)

result_1_py = A.dot(obs, return_numpy=True)
print(np.isclose(result_1, result_1_py).all())

result_2_py = A.dot(states, return_numpy=True)
result_2_py = result_2_py.astype("float64")[:, np.newaxis]
print(np.isclose(result_2, result_2_py).all())

result_3_py = A.dot(states, dims_to_omit=[0], return_numpy=True)
print(np.isclose(result_3, result_3_py).all())

# now try by putting obs and states into Categoricals themselves
obs = Categorical(values = mat_contents["o"])
states = Categorical(values = mat_contents["s"][0])

result_1_py_cat = A.dot(obs, return_numpy=True)
print(np.isclose(result_1,result_1_py_cat).all())

result_2_py_cat = A.dot(states, return_numpy=True)
result_2_py_cat = result_2_py_cat.astype("float64")[:, np.newaxis]
print(np.isclose(result_2, result_2_py_cat).all())

result_3_py_cat = A.dot(states, dims_to_omit=[0], return_numpy=True)
print(np.isclose(result_3_py_cat, result_3_py).all())
# %% 
# now do it with an AoA as the main matrix (the thing who's taking dot as a method)
A_both = np.empty(2,dtype=object)
A_both[0] = A.values
A_both[1] = A.values[0:2,:]
A_both[1] = A_both[1] / np.sum(A_both[1],axis = 0)

A_full = Categorical(values = A_both)

result = A_full.dot(states)
# %%


# %%
