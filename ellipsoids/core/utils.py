import jax.numpy as jnp

def sqrtm(S):

	lam, V = jnp.linalg.eigh(S)
	slam = jnp.sqrt(jnp.maximum(lam, 1e-6))

	if S.ndim == 2:
		return jnp.einsum("ik,jk,k->ij", V, V, slam)
	elif S.ndim == 3:
		return jnp.einsum("aik,ajk,ak->aij", V, V, slam)
	else:
		raise NotImplementedError()