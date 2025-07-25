import pytest
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax.scipy.special import digamma

from pymdp.maths import _exact_wnorm


@pytest.mark.parametrize("scale", [-12, -6, 0, 6, 12])
def test_exact_wnorm_finite(scale):
    """Ensure _exact_wnorm returns finite values and gradients across a wide range of magnitudes."""
    # Matrix whose entries span many orders of magnitude; include an explicit zero.
    base = jnp.array([
        [0.0, 2.0, 3.0, 4.0],
        [1.0, 5.0, 6.0, 0.5],
        [7.0, 8.0, 9.0, 10.0],
    ])
    A = base * (10.0 ** scale)

    # Forward computation
    wA = _exact_wnorm(A)
    assert jnp.all(jnp.isfinite(wA)), "_exact_wnorm produced NaN or Inf values"

    # Backward computation (gradient wrt A)
    grad_wA = grad(lambda x: _exact_wnorm(x).sum())(A)
    assert jnp.all(jnp.isfinite(grad_wA)), "Gradient of _exact_wnorm produced NaN or Inf values" 


def test_exact_wnorm_mathematical_correctness():
    """Test _exact_wnorm with a simple case where we can verify the mathematical result."""
    # Simple 2x2 case with known values
    A = jnp.array([[1.0, 2.0],
                   [3.0, 4.0]])
    
    result = _exact_wnorm(A)
    
    # Manually compute expected result using the formula
    # wA = log(A.sum(0)) - log(A) + 1/A - 1/A.sum(0) + digamma(A) - digamma(A.sum(0))
    A_sum = A.sum(axis=0)  # [4.0, 6.0]
    
    expected = (
        jnp.log(A_sum) - jnp.log(A) +
        1.0 / A - 1.0 / A_sum +
        jnp.array(digamma(A)) - jnp.array(digamma(A_sum))
    )
    expected = -expected  # minus sign in the implementation
    
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-8)
    
    # Verify output shape matches input
    assert result.shape == A.shape 


# -----------------------------------------------------------------------------
# Test 3: Higher Dirichlet counts → lower expected information gain
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 42])
def test_calc_pA_info_gain_precision(seed):
    """Scaling Dirichlet counts upward (more precise prior) should *reduce* the magnitude of (negative) information gain."""
    
    import jax.random as jr
    from pymdp.control import calc_pA_info_gain
    
    MINVAL = jnp.finfo(float).eps

    # ---------------------------
    # Shapes copied from T-Maze
    # ---------------------------
    num_obs = [5, 3, 3]   # observation dimensionalities per modality
    num_states = [5, 2]   # hidden-state factor sizes
    A_dependencies = [[0], [0, 1], [0, 1]]

    # Build a dummy (uniform) observation model just to get shapes correct
    A_like = [jnp.ones(shape) for shape in [
        (5, 5),           # modality 0: 5×5
        (3, 5, 2),        # modality 1: 3×5×2
        (3, 5, 2)         # modality 2: 3×5×2
    ]]

    key_root = jr.PRNGKey(seed)
    key_qs, key_qo, key_pA, key_scalar = jr.split(key_root, 4)

    # Random posterior over hidden states (qs)
    qs = []
    for n, k in zip(num_states, jr.split(key_qs, len(num_states))):
        probs = jr.uniform(k, (n,)) + MINVAL # ensure strictly positive
        qs.append(probs / probs.sum())

    # Random predictive beliefs over observations (qo)
    qo = []
    for n, k in zip(num_obs, jr.split(key_qo, len(num_obs))):
        probs = jr.uniform(k, (n,)) + MINVAL
        qo.append(probs / probs.sum())

    # Random Dirichlet base counts (>0)
    pA_uncertain = []
    for a_like, k in zip(A_like, jr.split(key_pA, len(A_like))):
        draw = jr.uniform(k, a_like.shape) * 2.0 + MINVAL  # counts in [MINVAL, MINVAL + 2.0)
        pA_uncertain.append(draw)

    # Slightly more precise prior
    scalar = 1.0 + jr.uniform(key_scalar, ()) * 0.1  # factor in (1.0, 1.1]
    pA_precise = [a * scalar for a in pA_uncertain]

    # Negative information gain for each prior
    neg_ig_uncertain = calc_pA_info_gain(pA_uncertain, qo, qs, A_dependencies)
    neg_ig_precise = calc_pA_info_gain(pA_precise, qo, qs, A_dependencies)

    # Remember: calc_pA_info_gain returns a *negative* quantity (due to historical sign convention).
    # A larger (less negative) value corresponds to *lower* expected information gain.
    assert neg_ig_precise > neg_ig_uncertain, "Increasing Dirichlet counts should make (negative) information gain less negative"
    
    