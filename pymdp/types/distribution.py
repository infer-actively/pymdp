import numpy as np


class Distribution(np.ndarray):
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0, strides=None, order=None):
        # Check if we are dealing and a multi-factor distribution
        dims = len(shape)
        factors = 1 if dims < 2 else shape[0]

        if factors > 2:
            obj = super().__new__(subtype, shape[0], object, buffer, offset, strides, order)
            # Ceate an ndarray for each for factor
            for factor in range(factors):
                dist_shape = (shape[1], shape[2])
                obj[factor] = super().__new__(
                    subtype, dist_shape, dtype, buffer, offset, strides, order
                )

        else:
            obj = super().__new__(subtype, shape, dtype, buffer, offset, strides, order)

        obj.factors = factors
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.factors = getattr(obj, "factors", 1)

    def dot(self):
        # Can override behavior based on ufuncs (true for all maths functions)
        raise NotImplementedError()


if __name__ == "__main__":

    dist_b = Distribution(shape=(6, 2, 1))
    print(dist_b.factors)
    print(dist_b[0] * 0)
    print(dist_b[0] + 10)
    print(np.dot(dist_b, dist_b))
