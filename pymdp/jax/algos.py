from maths import compute_likelihood

def run_vanilla_fpi(A, obs, prior):
    """ Vanilla fixed point iteration (jaxified) """

    likelihood = compute_likelihood(obs, A)

    pass