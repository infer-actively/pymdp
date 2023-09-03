import numpy as np


class Distribution(np.ndarray):
    def __new__(subtype, shapes, dtype=float, buffer=None, offset=0, strides=None, order=None):
        # Check if we are dealing and a multi-factor distribution

        n_factors = len(shapes) if isinstance(shapes[0], tuple) else 1

        if n_factors > 1:
            obj = super().__new__(subtype, n_factors, object, buffer, offset, strides, order)
            # Ceate an ndarray for each for factor
            for factor in range(n_factors):
                obj[factor] = super().__new__(
                    subtype, shapes[factor], dtype, buffer, offset, strides, order
                )

        else:
            obj = super().__new__(subtype, shapes, dtype, buffer, offset, strides, order)

        obj.n_factors = n_factors
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.n_factors = getattr(obj, "n_factors", 1)

    def dot(self):
        # Can override behavior based on ufuncs (true for all maths functions)
        raise NotImplementedError()


if __name__ == "__main__":
    
    # single factor example
    dist_b = Distribution(shapes=(6, 2, 1))
    print(f'Number of factors: {dist_b.n_factors}')
    print(dist_b * 0)
    print(dist_b + 10)
    # print(dist_b[0] * 0)
    # print(dist_b[0] + 10)
    # print(np.dot(dist_b, dist_b))

    # multi factor example
    dist_c = Distribution(shapes = ((6, 2, 1), (5, 3, 3)))
    print(f'Number of sub-arrays: {dist_c.n_factors}')
    for factor in range(dist_c.n_factors):
        print(f'Shape of sub-array {factor}: {dist_c[factor].shape}')

    print('I don''t understand why the values of the arrays, are what they are though...?\n')
    added_dist_c = dist_c + 10
    print(added_dist_c[0])
    print(added_dist_c[1])