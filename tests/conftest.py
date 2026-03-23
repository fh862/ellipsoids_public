# conftest.py — pytest session configuration
#
# jax.config.update must be called before any JAX computation is traced or
# compiled.  Placing it here at module level guarantees it runs before any
# test file is imported.

import jax

jax.config.update("jax_enable_x64", True)
