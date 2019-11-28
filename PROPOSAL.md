# Proposal

This living document charts the suggested `API` for the toolbox, feel free to edit as you see fit.

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

Some other considerations/ thoughts:
- Do not automatically normalize, as there may be some cases where this is not desired
- When passing dimensions, initialize as zero rather than randomly (this will allow users to first construct the distribution, then place non-zero values where they see fit)


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
