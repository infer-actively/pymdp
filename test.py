import numpy as np

from inferactively import Categorical, Dirichlet, test_function

# this allows testing against matlab spm_dot
from scipy.io import loadmat
import os 
import sys


if __name__ == "__main__":

    print("== Testing initialization via `dims` ===")

    A = Categorical(dims=4)
    print("Initalized `dims=4` gives shape {}".format(A.shape))

    A = Categorical(dims=4)
    print("Initalized `dims=4` gives shape {}".format(A.shape))

    A = Categorical(dims=[5, 4])
    print("Initalized `dims=[5,4]` gives shape {}".format(A.shape))

    A = Categorical(dims=[[5, 4], [4, 3]])
    print(
        "Initalized `dims=[[5,4], [4,3]]` gives shape {} while A[0] gives shape {}".format(
            A.shape, A[0].shape
        )
    )

    print("== Testing initialization via `values` ===")

    values = np.random.rand(5, 4)
    A = Categorical(values=values)
    print("Initalized `values=rand(5, 4)` gives shape {}".format(A.shape))

    values = np.random.rand(5)
    A = Categorical(values=values)
    print("Initalized `values=rand(5)` gives shape {}".format(A.shape))

    values_1 = np.random.rand(5, 4)
    values_2 = np.random.rand(4, 3)
    values = np.array([values_1, values_2])
    A = Categorical(values=values)
    print(
        "Initalized `values=(rand(5, 4), rand(4, 3))` gives shape {} where A[0] gives {}".format(
            A.shape, A[0].shape
        )
    )

    values_1 = np.random.rand(5)
    values_2 = np.random.rand(4)
    values = np.array([values_1, values_2])
    A = Categorical(values=values)
    print(
        "Initalized `values=(rand(5), rand(4))` gives shape {} where A[0] gives {}".format(
            A.shape, A[0].shape
        )
    )

    values_1 = np.random.rand(5)
    values_2 = np.random.rand(4, 3)
    values = np.array([values_1, values_2])
    A = Categorical(values=values)
    A.normalize()
    print(
        "Normalization Aoa: sum of (A[0]) gives {} and sum of (A[1]) gives {}".format(
            np.sum(A[0], axis=0), np.sum(A[1], axis=0)
        )
    )

    values = np.random.rand(5)
    A = Categorical(values=values)
    A.normalize()
    print(
        "Normalization test for single factor arrays: sum of A gives {}".format(
            np.sum(A.values, axis=0)
        )
    )

    # testing for the Categorical.dot() function

    # %% Case 1a - just vectors and matrices, discrete states/outcomes
    array_path = os.path.join(os.getcwd(),'matlab_tests/spm_dot/randomCase1a_dot.mat')

    mat_contents = loadmat(file_name=array_path)

    A_array = mat_contents['A']
    o_array = mat_contents['o']
    s = mat_contents['s']
    s_array = np.array(s,dtype=object)
    result1_matlab = mat_contents['result1']
    result2_matlab = mat_contents['result2']
    result3_matlab = mat_contents['result3']

    A = Categorical(values=A_array)

    print(o_array.shape)
    result1_python = A.dot(o_array, return_numpy = True)
    print(np.isclose(result1_matlab,result1_python).all()==True)

    result2_python = A.dot(s_array, return_numpy = True)
    print(np.isclose(result2_matlab,result2_python).all()==True)

    result3_python = A.dot(s_array, dims_to_omit = [1], return_numpy = True)
    print(np.isclose(result3_matlab,result3_python).all()==True)

    # %% Case 1b - continuous states and outcomes
    array_path = os.path.join(os.getcwd(),'matlab_tests/spm_dot/randomCase1b_dot.mat')

    # %%Case 2a: discrete states and outcomes, but add a third hidden state factor
    array_path = os.path.join(os.getcwd(),'matlab_tests/spm_dot/randomCase2a_dot.mat')

    # %% Case 2b: continuous states and outcomes, but add a third hidden state factor
    array_path = os.path.join(os.getcwd(),'matlab_tests/spm_dot/randomCase2b_dot.mat')

    # %% Case 3: continuous states and outcomes, but add a final (fourth) hidden state factor
    array_path = os.path.join(os.getcwd(),'matlab_tests/spm_dot/randomCase3_dot.mat')

