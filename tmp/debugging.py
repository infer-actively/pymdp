# %% IMPORTS
import os
import sys
import numpy as np
from scipy.io import loadmat

sys.path.append(".")
from inferactively import Categorical  # nopep8

# %% Debugging for Categorical.dot()
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
obs = Categorical(values=mat_contents["o"])
states = Categorical(values=mat_contents["s"][0])

result_1_py_cat = A.dot(obs, return_numpy=True)
print(np.isclose(result_1, result_1_py_cat).all())

result_2_py_cat = A.dot(states, return_numpy=True)
result_2_py_cat = result_2_py_cat.astype("float64")[:, np.newaxis]
print(np.isclose(result_2, result_2_py_cat).all())

result_3_py_cat = A.dot(states, dims_to_omit=[0], return_numpy=True)
print(np.isclose(result_3, result_3_py_cat).all())

# now do it with an AoA as the main matrix (the thing who's taking dot as a method)
A_both = np.empty(2, dtype=object)
A_both[0] = A.values
A_both[1] = A.values[0:2, :]
A_both[1] = A_both[1] / np.sum(A_both[1], axis=0)

A_full = Categorical(values=A_both)

result = A_full.dot(states)
print(
    "Resulting array is of shape {} with sub-shapes A[0].shape = {} and A[1].shape = {}".format(
        result.shape, result[0].shape, result[1].shape
    )
)
# %% Debugging for Categorical.cross()

# Case A (case 1a in the MATLAB script spm_cross_testing.m) - just outer-producting a vector with itself
array_path = os.path.join(os.getcwd(), "tests/data/cross_a.mat")
mat_contents = loadmat(file_name=array_path)
result_1 = mat_contents["result1"]
result_2 = mat_contents["result2"]

states = np.empty(1, dtype=object)
states[0] = mat_contents["s"][0, 0].squeeze()
states = Categorical(
    values=states
)  # this would create a 1-dimensional array of arrays (namely, self.IS_AOA == True)
result_1_py = states.cross()
print(np.isclose(result_1, result_1_py).all())

states = Categorical(
    values=mat_contents["s"][0, 0].squeeze()
)  # this creates a simple single-factor Categorical (namely, self.IS_AOA == False)
result_2_py = states.cross()
print(np.isclose(result_2, result_2_py).all())

# Case B (case 1b in the MATLAB script spm_cross_testing.m) - outer-producting two vectors together
array_path = os.path.join(os.getcwd(), "tests/data/cross_b.mat")
mat_contents = loadmat(file_name=array_path)
result_1 = mat_contents["result1"]
result_2 = mat_contents["result2"]

# first way, where both arrays as stored as two entries in a single AoA Categorical
states = Categorical(values=mat_contents["s"][0])
result_1_py = states.cross()
print(np.isclose(result_1, result_1_py).all())

# second way (type 1), where first array is a Categorical, second array is a straight numpy array
states_firstFactor = Categorical(values=mat_contents["s"][0][0])
states_secondFactor = mat_contents["s"][0][1]
result_2a_py = states_firstFactor.cross(states_secondFactor)
print(np.isclose(result_2, result_2a_py).all())

# second way (type 2), where first array is a Categorical, second array is another Categorical
states_firstFactor = Categorical(values=mat_contents["s"][0][0])
states_secondFactor = Categorical(values=mat_contents["s"][0][1])
result_2b_py = states_firstFactor.cross(states_secondFactor)
print(np.isclose(result_2, result_2b_py).all())

# Case C (case 2a in the MATLAB script spm_cross_testing.m) - outer producting a vector and a matrix
array_path = os.path.join(os.getcwd(), "tests/data/cross_c.mat")
mat_contents = loadmat(file_name=array_path)
result_1 = mat_contents["result1"]
random_vec = Categorical(values=mat_contents["random_vec"])

# first way, where first array is a Categorical, second array is a numpy ndarray
random_matrix = mat_contents["random_matrix"]
result_1a_py = random_vec.cross(random_matrix)
print(np.isclose(result_1, result_1a_py).all())

# second way, where first array is a Categorical, second array is a Categorical
random_matrix = Categorical(values=mat_contents["random_matrix"])
result_1b_py = random_vec.cross(random_matrix)
print(np.isclose(result_1, result_1b_py).all())

# Case D (case 2b in the MATLAB script spm_cross_testing.m) - outer producting a vector and a sequence of vectors
array_path = os.path.join(os.getcwd(), "tests/data/cross_d.mat")
mat_contents = loadmat(file_name=array_path)
result_1 = mat_contents["result1"]
random_vec = Categorical(values=mat_contents["random_vec"])
states = mat_contents["s"]
for i in range(len(states)):
    states[i] = states[i].squeeze()

# first way, where first array is a Categorical, second array is a numpy ndarray (dtype = object)
result_1a_py = random_vec.cross(states)
print(np.isclose(result_1, result_1a_py).all())

# second way, where first array is a Categorical, second array is a Categorical (where self.IS_AOA = True)
states = Categorical(values=states[0])
result_1b_py = random_vec.cross(states)
print(np.isclose(result_1, result_1b_py).all())

# Case E (case 3a in the MATLAB script spm_cross_testing.m) - outer producting two sequences of vectors
array_path = os.path.join(os.getcwd(), "tests/data/cross_e.mat")
mat_contents = loadmat(file_name=array_path)
result_1 = mat_contents["result1"]

s2 = mat_contents["s2"]
s2_new = np.empty(s2.shape[1], dtype=object)
for i in range(len(s2_new)):
    s2_new[i] = s2[0][i].squeeze()

# first way (type 1), first sequence is a Categorical (self.AOA = True) and second sequence is a numpy ndarray (dtype = object)
s1 = Categorical(values=mat_contents["s1"][0])
result_1aa_py = s1.cross(s2_new)
print(np.isclose(result_1, result_1aa_py).all())

# first way (type 2), first sequence is a Categorical (self.AOA = False) and second sequence is a numpy ndarray (dtype = object)
s1 = Categorical(values=mat_contents["s1"][0][0])
result_1ab_py = s1.cross(s2_new)
print(np.isclose(result_1, result_1ab_py).all())

s2_new = Categorical(values=mat_contents["s2"][0])
# second way (type 1), first sequence is a Categorical (self.AOA = True) and second sequence is a Categorical
s1 = Categorical(values=mat_contents["s1"][0])
result_2aa_py = s1.cross(s2_new)
print(np.isclose(result_1, result_2aa_py).all())

# second way (type 2), first sequence is a Categorical (self.AOA = False) and second sequence is a Categorical
s1 = Categorical(values=mat_contents["s1"][0][0])
result_2ab_py = s1.cross(s2_new)
print(np.isclose(result_1, result_2ab_py).all())


# %%
