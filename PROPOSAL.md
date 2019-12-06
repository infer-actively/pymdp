# Proposal

This file contains an overview of the MDP toolbox. Please refer to items marked `@TODO` for actionable tasks. 

## Table of contents
1. [`Categorical`](#categorical)
2. [`Dirichlet`](#dirichlet)
3. [`Functions`](#functions)
4. [`Examples`](#examples)


# `Categorical` <a name="categorical"></a>

This class represents a [Categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution). At its core, it is a wrapper around `numpy` arrays and matrices, and serves to provide useful probabilistic functions, manipulations, as well as to manage details such as shape and error checking. 

The following list only contains the public-facing interface. Please refer to the source code for a full list of functions.

## `init`
*status*: Done ✓

*Note*: Could be worth filling out the docs to explain the `arrays of arrays` formulation in depth

```python
def __init__(self, dims=None, values=None):
    """Initialize a Categorical distribution

    Parameters
    ----------
    `dims` [list :: int] || [list :: list] 
        Specifies the number and size of dimensions
    `values` [np.ndarray]
        The parameters of the distribution
    """
```

## `dot`
*status*: `@TODO:` Need to move and elaborate @conor's documentation to the function and provide inline examples. While this function has been tested extensively against the `SPM` implementation (see the `tests` folder), it has not been tested in the wild. 

*Note*: This function is a wrapper for `F.spm_dot`, which contains the core implementation.

```python
def __init__(self, dims=None, values=None):
    """ Dot product of a Categorical distribution with `x`

    The dimensions in `dims_to_omit` will not be summed across during the dot product

    Parameters
    ----------
    `x` [1D numpy.ndarray] || [Categorical]
        The alternative array to perform the dot product with
    `dims_to_omit` [list :: int] (optional)
        Which dimensions to omit
    `return_numpy` [bool] (optional)
        Whether to return `np.ndarray` or `Categorical`
    """
```

## `cross`
*status*: `@TODO:` Need to move and elaborate @conor's documentation to the function and provide inline examples. While this function has been tested extensively against the `SPM` implementation (see the `tests` folder), it has not been tested in the wild. 

*Note*: This function is a wrapper for `F.spm_cross`, which contains the core implementation.

```python
def __init__(self, dims=None, values=None):
    """ Multi-dimensional outer product

    If no `x` argument is passed, the function returns the "auto-outer product" of self

    Otherwise, the function will recursively take the outer product of the initial entry
    of `x` with `self` until it has depleted the possible entries of `x` that it can outer-product

    Parameters
    ----------
    `x` [np.ndarray] || [Categorical] (optional)
        The values to perform the outer-product with
    `args` [np.ndarray] || [Categorical] (optional)
        Perform the outer product of the `args` with self
    
    Returns
    -------
    `y` [np.ndarray] || [Categorical]
        The result of the outer-product

    """
```

## `normalize`
*status*: `@TODO:` Should move this implementation to the `functions` file, and simply call `F.normalize(self.values)` (or perform some loop in the case of arrays of array). 

```python
def normalize(self):
    """ Normalize distribution

    This function will ensure the distribution(s) integrate to 1.0
    In the case `ndims` >= 2, normalization is performed along the columns of the arrays

    """
```

## `is_normalized`
*status*: Done ✓

```python
def is_normalized(self):
    """ Checks whether columns sum to 1

    Note this operates within some margin of error (10^-4)

    """
```

## `remove_zeros`
*status*: Done ✓

```python
def remove_zeros(self):
    """ Remove zeros by adding a small number

    This function avoids division by zero
    exp(-16) is used as the minimum value

    """
```

## `contains_zeros`
*status*: Done ✓

```python
def contains_zeros(self):
    """ Checks if any values are zero

    Returns
    ----------
    bool
        Whether there are any zeros

    """
```

# `Dirichlet` <a name="categorical"></a>

# `Functions` <a name="categorical"></a>

# `Examples` <a name="categorical"></a>