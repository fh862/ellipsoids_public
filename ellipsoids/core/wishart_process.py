import jax
import jax.numpy as jnp
from . import chebyshev


class WishartProcessModel:

    def __init__(
            self, degree, num_dims, extra_dims, variance_scale,
            decay_rate, diag_term, num_dims_cov = None
        ):
        """
        degree : int
            Degree of the Chebyshev polynomial. The total number of
            basis functions is (degree ** num_dims)

        num_dims : int
            Number of stimulus dimensions.

        extra_dims : int
            Number of additional dimensions used for the Wishart
            process ("degrees of freedom").

        variance_scale : float
            Scale parameter for the prior variance of the zeroth-order
            (constant) basis coefficient. Higher-order coefficients are
            down-weighted geometrically.
            
            Corresponds to the parameter γ (gamma) in the eLife paper.
    
        decay_rate : float
            Geometric decay factor controlling how strongly higher-order
            polynomial coefficients are weighted in the prior. Specifically,
            the prior variance of a coefficient of total polynomial order d
            is given by:
            
                variance_scale × (decay_rate ** d)
            
            Corresponds to the parameter ε (epsilon) in the eLife paper.
            Note that smaller values of `decay_rate` impose stronger
            suppression of higher-order basis functions (yielding smoother 
            covariance fields), whereas larger values allow greater contribution 
            from higher-order basis functions. Although the name `decay_rate` 
            can be counterintuitive in this respect, it is retained for consistency
            with existing code.

        diag_term : float
            Minimum variance for the ellipsoids.
            
        num_dims_cov : int
            Number of covariance matrix dimensions. By default, this is set
            equal to the stimulus dimensionality (`num_dims`). In some cases,
            however, these may differ. For example, the stimulus may be defined
            in a 3D space where the third dimension represents a contextual or
            experimental variable (e.g., eccentricity, spatial frequency, or
            adaptation state) along which the psychometric field varies smoothly.
            In such cases, the perceptual noise may still be well described by
            a 2D covariance matrix, even though the stimulus space itself is 3D.
        
        """

        if num_dims not in (2, 3):
            raise ValueError("`num_dims` must be equal to 2 or 3.")
            
        # If not provided, use the stimulus dimensionality
        if num_dims_cov is None:
            num_dims_cov = num_dims
            
        # We don't allow a higher-dimensional covariance than the stimulus itself.
        if num_dims_cov > num_dims:
            raise ValueError(
                f"`num_dims_cov` ({num_dims_cov}) cannot be larger than `num_dims` ({num_dims}). "
                "The covariance dimensionality must be less than or equal to the stimulus dimensionality."
            )

        self.degree = degree
        self.num_dims = num_dims
        self._num_dims_cov = num_dims_cov
        self.extra_dims = extra_dims
        self.variance_scale = variance_scale
        self.decay_rate = decay_rate
        self.diag_term = diag_term
        self.cheb_basis = chebyshev.chebyshev_basis(degree)

        if self.num_dims == 2:
            basis_degrees = (
                jnp.arange(self.degree)[:, None] +
                jnp.arange(self.degree)[None, :]
            )

        elif self.num_dims == 3:
            basis_degrees = (
                jnp.arange(self.degree)[:, None, None] +
                jnp.arange(self.degree)[None, :, None] + 
                jnp.arange(self.degree)[None, None, :]
            )

        self.W_prior_variances = (
            self.variance_scale * (self.decay_rate ** basis_degrees)
        )
        
    @property
    def num_dims_cov(self):
        """
        Dimensionality of the covariance matrices.

        For older pickled models that predate `num_dims_cov`, this
        falls back to `self.num_dims`.
        """
        return getattr(self, "_num_dims_cov", self.num_dims)

    @num_dims_cov.setter
    def num_dims_cov(self, value):
        self._num_dims_cov = value

    def sample_W_prior(self, key):
        """
        Draw a sample from the prior over basis function coefficients.
    
        Parameters
        ----------
        key : jax.random.PRNGKey
            Seed for random number generator.
    
        Returns
        -------
        W : array
            Array holding the coefficients sampled from the prior.
            Shape: (*basis_shape, num_dims_cov, num_dims_cov + extra_dims)
            where basis_shape is (degree, degree) for 2D or
            (degree, degree, degree) for 3D.
        """
        variances = self.W_prior_variances  # shape: (deg, deg) or (deg, deg, deg)
    
        # Add two singleton axes at the end, works for both 2D and 3D:
        #   (deg, deg)        -> (deg, deg, 1, 1)
        #   (deg, deg, deg)   -> (deg, deg, deg, 1, 1)
        scale = jnp.sqrt(variances)[..., None, None]
    
        noise = jax.random.normal(
            key,
            shape=(*variances.shape, self.num_dims_cov, self.num_dims_cov + self.extra_dims),
        )
    
        return scale * noise

    def logprior_density_W(self, W):
        """
        Parameters
        ----------
        W : jax.numpy.array
            Array of coefficients for the Chebyshev basis. If
            num_dims == 2, then W.shape should be
            (degree, degree, num_dims, num_dims + extra_dims).
            If num_dims == 3, then W.shape should be
            (degree, degree, degree, num_dims, num_dims + extra_dims).

        Returns
        -------
        logdensity : float
            Log density of the prior distribution over coefficients.
        """
        variances = self.W_prior_variances  # shape: (deg, deg) or (deg, deg, deg)
    
        # Broadcast scale over the last two coefficient axes, regardless of 2D or 3D:
        #   (deg, deg)       -> (deg, deg, 1, 1)
        #   (deg, deg, deg)  -> (deg, deg, deg, 1, 1)
        scale = jnp.sqrt(variances)[..., None, None]
    
        return jax.scipy.stats.norm.logpdf(W, scale=scale).sum()

    def compute_U(self, W, x):
        """
        Parameters
        ----------
        W : jax.numpy.array
            Array of coefficients for the Chebyshev basis. If
            num_dims == 2, then W.shape should be
            (degree, degree, num_dims, num_dims + extra_dims).
            If num_dims == 3, then W.shape should be
            (degree, degree, degree, num_dims, num_dims + extra_dims).

        x : jax.numpy.array
            Array of points to evaluate the linear combination of
            basis functions. Has shape (..., num_dims)

        Returns
        -------
        U : jax.numpy.array
            Array with shape (..., num_dims, num_dims + extra_dims).
        """
        assert x.shape[-1] == self.num_dims

        # xt.shape = (N, num_dims)
        xt = jnp.clip(x.reshape(-1, x.shape[-1]), -1.0, 1.0)

        if self.num_dims == 2:
            # phi.shape = (N, degree, degree)
            phi = (
                chebyshev.evaluate(self.cheb_basis, xt[:, 0])[:, :, None] *
                chebyshev.evaluate(self.cheb_basis, xt[:, 1])[:, None, :]
            )
            U = jnp.einsum("abdv,iab->idv", W, phi)

        elif self.num_dims == 3:
            # phi.shape = (N, degree, degree, degree)
            phi = (
                chebyshev.evaluate(self.cheb_basis, xt[:, 0])[:, :, None, None] *
                chebyshev.evaluate(self.cheb_basis, xt[:, 1])[:, None, :, None] *
                chebyshev.evaluate(self.cheb_basis, xt[:, 2])[:, None, None, :]
            )
            U = jnp.einsum("abcdv,iabc->idv", W, phi)

        return U.reshape(*x.shape[:-1], U.shape[-2], U.shape[-1])

    def compute_Sigmas(self, U):
        """
        Parameters
        ----------
        U : jax.numpy.array
            Array with shape (..., num_dims, num_dims + extra_dims).

        Returns
        -------
        Sigmas : jax.numpy.array
            Array with shape (..., num_dims, num_dims). Each of 2d
            slice of this array along the final two dimensions
            is a positive definite matrix specifying an ellipsoid.
        """

        # Reshape U, flattening all but the last two dimensions.
        shp = U.shape[:-2]
        U = U.reshape(-1, *U.shape[-2:])

        # Compute covariance matrix
        S = (
            jnp.einsum("ijk,ihk->ijh", U, U) +
            self.diag_term * jnp.eye(self.num_dims_cov)[None, :, :]
        )

        # Reshape to original dimensions.
        return S.reshape(*shp, self.num_dims_cov, self.num_dims_cov)
