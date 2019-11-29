import numpy as np

from inferactively import Categorical, Dirichlet, test_function


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
        "Normalization test for AoAs: sum of first array A (A[0]) gives {} and sum of second array (A[1]) gives {}".format(
            np.sum(A[0], axis = 0), np.sum(A[1], axis = 0)
        )
    )

    values = np.random.rand(5)
    A = Categorical(values=values)
    A.normalize()
    print(
        "Normalization test for single factor arrays: sum of A gives {}".format(
            np.sum(A.values, axis = 0)
        )
    )
