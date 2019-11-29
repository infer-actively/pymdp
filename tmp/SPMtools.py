"""

Python (read: numpy-based) versions of common SPM functions

@author: conor
"""

import numpy as np
import scipy as sp

# %% 
def spm_dot(X,x,dims2omit=None):
    
    """
    Multi-dimensional dot product, based on the spm_dot.m function from the DEM toolbox in SPM for MATLAB
    ARGUMENTS:
        X : multi-dimensional array (numpy ndarray) such as a a likelihood mapping, used to generate predictive distributions via likelihoods or to perform model inversion (inverting the likelihood model to get marginal likelihoods)
        x : numpy 'array of arrays' or a np.array(dtype=object) type of ndarray. Represents sets of (multi-modality or multi-factorial) categorical distributions or samples
        dims2omit: a list of integers specifying the dimensions of X (the indices of the entries of x) to be omitted during the dot product - namely, these dimensions will not be marginalized out via summation 
        and the result of the dot product will maintain dimensionality along those dimensions specified herein
    RETURNS:
        X : the result of the multidimensional dot product, either a 1d ndarray or a multi-dimensional ndarray
    """ 
       
    # initialize dimensions
    if x.dtype == 'object':
        DIM = (np.arange(0,len(x)) + np.ndim(X) - len(x)).astype(int)
    else:
        DIM = np.array([0],dtype=int)
        x_new = np.empty(1,dtype=object)
        x_new[0] = x
        x = x_new
        
    if dims2omit is not None:
       if not isinstance(dims2omit,list):
           raise ValueError("dims2omit must be a list!")
       DIM = np.delete(DIM,dims2omit)
       if len(x) == 1:
           x = np.empty([0],dtype=object) # because of stupid stuff regarding deleting dimensions from numpy arrays-of-arrays
       else:
           x = np.delete(x,dims2omit)  
            
    # inner product using recursive summation and implicit expansion ('broadcasting' in numpy lingo)
    for d in range(len(x)):
        s         = np.ones(np.ndim(X),dtype=int)
        s[DIM[d]] = np.shape(x[d])[0]
        X         = X * x[d].reshape(tuple(s))
        X         = np.sum(X,axis=DIM[d],keepdims=True)
        
    X = np.squeeze(X)
    
    return X

# %%
def spm_cross(X, x=None, *args):
    """
    Function that returns the "spm_cross" of either two vectors, 
    two numpy array objecgts, one of each, or simply every element within a 
    single object. It appears that this is just a sophisticated way to do
    an outer product of two arbitrarily sized array objects.
    
    Arguments
    ----------
    X (numpy ndarray, array-object):
        Vector or object of vectors that you want to cross product with either
        itself or with x. If X is a vector and x is an object, it iteratively
        spans through the elements of x, increasing the size of the output
        vector, until it has reached maximum capcaity.
    
    x (numpy ndarray, array-object, or None):
        If None, the function simply returns the "self outer product" of X.
        Otherwise, it will iteratively / recursively outer product the initial entry
        of x with the ongoing result until it has depleted the entries of x to take the
        outer product with respect to.
        
    args (numpy ndarray, array-object, None):
        If an nd.array or array-object is passed into the function, it will
        take the outer product of the ndarray or the first entry of the array-object (respectively)
        with X. Otherwise, if this is None, then the result will simply be spm_cross(X,x)
    
    Returns
    -------
    Y (numpy ndarray):
        the cross-producted vector between X and the remaining arguments to the function.
    """

    if len(args) == 0 and x is None:  
        if X.dtype == 'object':
            Y = spm_cross(*list(X))
            
        elif np.issubdtype(X.dtype, np.number):
            Y = X
            
        return Y

    if X.dtype == 'object':
        X = spm_cross(*list(X))
        
    if x is not None and x.dtype == 'object':
        x = spm_cross(*list(x))

    reshape_dims = tuple(list(X.shape) + list(np.ones(x.ndim, dtype=int)))
    A = X.reshape(reshape_dims)
    
    reshape_dims = tuple(list(np.ones(X.ndim, dtype=int)) + list(x.shape))
    B = x.reshape(reshape_dims)

    Y = np.squeeze(A * B)

    for x in args:
        Y = spm_cross(Y, x)

    return Y

# %% 
def spm_MDP_G(A, x):
    """
    Calculates the Bayesian surprise in the same way as spm_MDP_G.m does in 
    the original matlab code.
    
    Arguments
    ----------
    A (numpy ndarray or array-object):
        array assigning likelihoods of observations/outcomes under the various hidden state configurations
    
    x (numpy ndarray or array-object):
        Categorical distribution presenting probabilities of hidden states (this can also be interpreted as the 
        predictive density over hidden states/causes if you're calculating the 
        expected Bayesian surprise)
        
    Returns
    -------
    G (float):
        the (expected or not) Bayesian surprise under the density specified by x --
        namely, this scores how much an expected observation would update hidden states
        x, were it observed. 
    """
    
    Ng = A.shape

    # start = dt.datetime.now()
    # probability distribution over the hidden causes: i.e., Q(x)
    qx = spm_cross(x)
    G = 0
    qo = 0
    idx = np.array(np.where(qx > np.exp(-16))).T
    # print('Time taken to cross product the states and find non-zero indices: %1.2f' %(dt.datetime.now() - start).total_seconds())

    # start_iterative = dt.datetime.now()
    # accumulate expectation of entropy: i.e., E[lnP(o|x)]
    for i in idx:

        # probability over outcomes for this combination of causes
        po = np.ones(1)

        # start = dt.datetime.now()
        # for g1 in range(Ng[0]):
        #     for g2 in range(Ng[1]):
        #         index_vector = [slice(0, A[g1, g2].shape[0])] + list(i)
        #         po = spm_cross(po, A[g1, g2][tuple(index_vector)])
        # print('Time taken to cross product po with the likelihood: %1.2f' %(dt.datetime.now() - start).total_seconds())

        # This version enforces our prior assumption that there is no information gain about states to be gleaned from the reward observation modalities, or from the first/last location modalities
        # start = dt.datetime.now()
        g1 = 0
        for g2 in range(1, A.shape[1]-1 ): # this is the prior
            index_vector = [slice(0, A[g1,g2].shape[0])] + list(i)
            po = spm_cross(po, A[g1,g2][tuple(index_vector)])
        # print('Time taken to cross product po with the likelihood: %1.2f' %(dt.datetime.now() - start).total_seconds())

        # start = dt.datetime.now()
        po = po.ravel()
        # print('Time taken to unravel po: %1.2f' %(dt.datetime.now() - start).total_seconds())

        # start = dt.datetime.now()
        qo += qx[tuple(i)] * po
        # print('Time taken element-wise multiple qx with po and add to qo: %1.2f' %(dt.datetime.now() - start).total_seconds())

        # start = dt.datetime.now()
        G += qx[tuple(i)] * po.dot(np.log(po + np.exp(-16)))
        # print('Time taken element-wise multiple qx with entropy of po and add to G: %1.2f' %(dt.datetime.now() - start).total_seconds())

    # print('Time taken to do the iterative surprise calculation: %1.2f' %(dt.datetime.now() - start_iterative).total_seconds())

    # start = dt.datetime.now()
    # subtract negative entropy of expectations: i.e., E[lnQ(o)]
    # (note the double negative in the equation)
    G  = G - qo.dot(np.log(qo + np.exp(-16)))
    # print('Time taken to substract the negative entropy of expectations: %1.2f' %(dt.datetime.now() - start).total_seconds())

    return G
# %%
def spm_norm(A):
    """ 
    Normalization of a transition probability matrix or likelihood mapping from states to observations
    """
    
    column_sums = np.sum(A, axis=0)
    A = np.divide(A, column_sums)
    nan_idx = np.where(np.isnan(A))
    if np.array(nan_idx).size != 0:
        A[np.where(np.isnan(A))] = np.divide(1.0,np.sum(A,axis=0))
    return A
    
# %% 

def spm_wnorm(A):
    """
    Normalization of a prior over Dirichlet parameters, used in updates for information gain
    """
    
    A = A + 1e-16
    
    norm = np.divide(1.0, np.sum(A,axis=0))
    
    avg = np.divide(1.0, A)
    
    wA = norm - avg
   
    return wA

# %% 

def spm_betaln(z):
    """
     returns the log the multivariate beta function of a vector.
    FORMAT y = spm_betaln(z)
     y = spm_betaln(z) computes the natural logarithm of the beta function
     for corresponding elements of the vector z. if concerned is a matrix,
     the logarithm is taken over the columns of the matrix z.
    """

    y     = np.sum(sp.special.gammaln(z),axis=0) - sp.special.gammaln(np.sum(z,axis=0))

    return y

# %%

def spm_psi(A):

    """
    Normalized psi function for an array
    """

    column_sum = sp.special.psi(np.sum(A,axis=0))
    A = sp.special.psi(A) - column_sum

    return A
# %%

def spm_KL_dir(q,p):

    """KL divergence between two Dirichlet distributions
    FORMAT [d] = spm_kl_dirichlet(lambda_q,lambda_p)

    Calculate KL(Q||P) = <log Q/P> where avg is wrt Q between two Dirichlet 
    distributions Q and P

    lambda_q   -   concentration parameter matrix of Q
    lambda_p   -   concentration parameter matrix of P

    This routine uses an efficient computation that handles arrays, matrices 
    or vectors. It returns the sum of divergences over columns.
    """

    d = spm_betaln(p) - spm_betaln(q) - np.sum((p - q) * spm_psi(q + 1/32),axis=0)
    d = np.sum(d.ravel())

    return d
