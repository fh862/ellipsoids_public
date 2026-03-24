import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp

# Example usage
# -------------
#
# >> degree = 5 
# >> x = jnp.linspace(-1, 1, 100)
# >> basis_coeffs = chebyshev_basis(degree)
# >> evaluate(basis_coefficients, x)

#evaluate = jax.vmap(jnp.polyval, (0, None), 1)
def evaluate(basis_coefficients, x):
    return jax.vmap(jnp.polyval, (0, None), 1)(basis_coefficients, x)

def chebyshev_basis(degree):
    """
    Create a Chebyshev polynomial basis in 1D.

    Parameters
    ----------
    degree : int
        Degree of the polynomial basis.

    Returns
    --------
    basis_coefficients : array
        Square matrix. Each successive row specifies
        the coefficents of the Chebyshev polynomial
        sequence.
    """

    twox = np.zeros(degree)
    twox[-2] = 2.0

    T = np.zeros((degree, degree))
    T[0, -1] = 1.0
    T[1, -2] = 1.0
    for n in range(1, degree - 1):
        T[n + 1] = np.polyadd(np.polymul(twox, T[n]), -1 * T[n - 1])

    return jnp.array(T)

