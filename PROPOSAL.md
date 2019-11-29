# Proposal

This living document charts the suggested `API` for the toolbox, feel free to edit as you see fit

## Initialization 

Users should have two methods for initializing distributions. The first is to pass values directly, and the second is to pass the proposed dimensions, which are then use to construct zero-filled arrays of the same dimensions. 

```
values = np.array([[0.5, 0.6], [0.3, 0.7]])
A = Categorical(values=values)
```

```
A = Categorical(dims=[5, 4])
```

For multi-factor implementations, the same principe applies. We will perform checks to make sure that, when using values, users pass `numpy` arrays of arrays. For instance:

`@comment: is there anyway we can be more flexible and recast to numpy arrays-of-arrays?`

```
values = np.array(2, dtype=object)
values[0] = np.array([[0.5, 0.6], [0.3, 0.7]])
values[1] = np.array([[0.1, 0.4], [0.1, 0.7]])
A = Categorical(values=values)
```

```
A = Categorical(dims=[[5, 4], [4, 3]])
```

In the case that users implement an array-of-array distribution, we should set a flag on the object (i.e. `IS_AOA = True`), so that we can deal with it properly in functions.


## Inference

Suggestion of an `update_posterior` function with the following `API`:

```
def update_posterior(observations, A, Qs, prior= None, n_iters=5):
    
    joint_likelihood = np.ones(tuple(n_states))

    # get joint likelihood
    for j in range(n_modalities):
        joint_likelihood *= spm_dot(A[j], observations[j])

    # fixed point iteration
    for _ in range(n_iters):
        for f in range(n_factors):

            marginal_likelihood = spm_dot(joint_likelihood, Qs, omit_dims=[f])
            if prior is not None:
                # add prior
            else:
                Qs[f] = softmax(log(marginal_likelihood))
    return Qs
```

Some considerations:
- Is it possible to take observations as indexes as well as row vectors?
- Will this work with single factors?
- Should we keep calculation of prior separate

## Conor's notes regarding implementing the Categorical class

- `spm_dot` as a method of `Categorical()` *AND* a standalone function
    - If this is the case, we need to make sure that when you call it as a method, you specify the modality you're dotting as well as the hidden state factors you're dotting it with (and the `dims2omit` list as usual)
    - Example: If our `Categorical()` is called `A_gp`, and we're dotting one of its modalities (**i.e. we're dotting one of the various likelihoods mappings stored within it**) with some hidden states `Qs`, we need to do something like:
        ```
        A_gp.dot(g, Qs, dims2omit)
        ```
        where `g` indexes a particular modality
- Some notes on doing `spm_norm` and `spm_wnorm` on different kinds of `Categorical()` instances:
    - just make sure the `np.sum(ndarray, axis = 0)` works in the intended way on row vectors (1d nd-arrays), and sums across the row, even though those are technically 'columns'


