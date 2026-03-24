import jax.numpy as jnp

from . import utils

# Constant used to plot ellipses
_THETAS = jnp.linspace(0, 2 * jnp.pi, 50)
_SINUSOIDS = jnp.vstack(
	(jnp.cos(_THETAS), jnp.sin(_THETAS))
)


def plot_ellipse(ax, mu, S, **kwargs):
	"""
	Plot an ellipse to visualize a bivariate normal distribution.

	Parameters
	----------
	ax : matplotlib.pyplot.Axes object
		Axis object which will plot the ellipse.

	mu : array
		Vector with two elements specifying x and y
		coordinate of the mean.

	S : array
		Covariance matrix.

	**kwargs : dict
		Additional keyword arguments are passed to ax.plot(...)

	Returns
	-------
	ellipse : matplotlib.pyplot.Line2D
		Line objects representing the plotted ellipse.
	"""
	_x, _y = utils.sqrtm(S) @ _SINUSOIDS
	return ax.plot(mu[0] + _x, mu[1] + _y, **kwargs)

