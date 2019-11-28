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
