"""
Geodesics between two fixed points (boundary value problem), built on the
initial value solver in `euclidization.geodesics`.

`euclidization.geodesics` integrates the geodesic equation forward from a
start point x0 and an initial velocity v0 (`exp_map`, `geodesic_path`,
`geodesic_length`). This module adds the shooting method on top of it: given
x0 and an end point x1, search for a v0 whose geodesic ends at x1.

Every function takes the inverse metric g_inv(x) -> (d, d), i.e. the Wishart
covariance Sigma(x), the same convention as `euclidization.geodesics`. For a
fitted Wishart model:

    g_inv = lambda x: model.compute_Sigmas(model.compute_U(W, x))

This module replaces `core/geodesics_archived.py`, which takes the precision
matrix P(x) = Sigma(x)^{-1} instead. `euclidization` is not yet listed in
pyproject.toml; it must be importable (e.g. installed from its repo).
"""

import jax
import jax.numpy as jnp
import optimistix
from evosax.algorithms import CMA_ES
from euclidization import geodesics as euclid_geodesics
from optimistix import LevenbergMarquardt

#%% 1. Shooting method: find the initial velocity
def shoot_geodesics(
        g_inv, x0s, x1s, keys, *,
        num_generations=20, popsize=20, tolerance=1e-3,
        v0_inits=None, n_steps=512,
    ):
    """
    For each pair of points, find an initial velocity v0 whose geodesic from
    x0 ends at x1 (shooting method).

    For each pair, an evolutionary strategy (CMA-ES) searches over v0. Each
    candidate is shot forward with `euclid_geodesics.exp_map`, and its fitness
    is the Euclidean distance between the geodesic's end point x(1) and x1.
    The search for a pair stops after `num_generations`, or as soon as its
    best fitness falls below `tolerance`.

    All pairs run in one compiled, vmapped `jax.lax.while_loop`. Under vmap
    the loop keeps running while any pair is unfinished, but the state of
    pairs that have finished is left unchanged. Each pair therefore keeps its
    own stop, and its result does not depend on the other pairs in the batch.
    For a single pair, pass arrays with a leading axis of size 1
    (x0[None], x1[None], key[None]).

    The fitness measures only how close the geodesic lands to x1, not its
    length. If several geodesics join x0 and x1 (e.g. the short and the long
    arc of a great circle on a sphere), the search returns whichever one it
    converges to, not necessarily the shortest. The default start x1 - x0
    favors the geodesic closest to the straight line, which is usually the
    shortest.

    Used for geodesic distances between reference and comparison stimuli,
    one pair at a time (e.g. geodist/compute_geodesics_new.py,
    analysis/MOCS_thresholds.py) or many noisy samples at once
    (e.g. geodist/eval_geodesic_PMF.py).

    Parameters
    ----------
    g_inv : Function R^d -> R^(d,d) returning the inverse metric g^{ij}(x),
        i.e. the covariance Sigma(x).
    x0s : array, shape (B, d)
        Start points.
    x1s : array, shape (B, d)
        End points.
    keys : array, shape (B, 2)
        One random key per pair for CMA-ES sampling, e.g.
        jax.random.split(key, B).
    num_generations : int
        Maximum number of CMA-ES generations per pair.
    popsize : int
        Number of candidate velocities per generation.
    tolerance : float
        Stop a pair once its best end-point error is below this value (same
        units as x).
    v0_inits : array, shape (B, d), optional
        Starting means of the CMA-ES searches (initial guesses, not the
        answer). Defaults to x1s - x0s, which is the exact answer when the
        metric is flat (the geodesic is then the straight line
        x0 + t (x1 - x0)).
    n_steps : int
        Number of fixed integration steps per shot, passed to `exp_map`. The
        default, 512, is as accurate as the old 1000-step Dopri5 solver in
        `core/geodesics_archived.py` (end-point error ~1e-13 on the sub1 Wishart fits).
        Fewer steps are faster (time scales with n_steps) and still accurate
        well below `tolerance`: about 5e-7 at 32 steps, 1e-8 at 64.

    Returns
    -------
    state : evosax CMA-ES states, each field with a leading axis of size B.
    metrics : dict, each entry with a leading axis of size B
        metrics["best_solution"] is the best v0 found, with shape (B, d);
        metrics["best_fitness"] is its end-point error, shape (B,);
        metrics["generation_counter"] is the number of generations run, shape (B,).
    """
    if v0_inits is None:
        v0_inits = x1s - x0s

    # The CMA-ES object only needs the shape of v0; each search starts at its
    # v0_init, which is passed to strategy.init below.
    strategy = CMA_ES(population_size=popsize, solution=jnp.zeros(x0s.shape[-1]))
    es_params = strategy.default_params

    def shoot_one_pair(x0, x1, key, v0_init):
        # End-point error of every candidate v0 in a population, shape (popsize,).
        def endpoint_error(v0_pop):
            x1_pop = jax.vmap(
                lambda v0: euclid_geodesics.exp_map(g_inv, x0, v0, n_steps)
            )(v0_pop)
            return jnp.linalg.norm(x1_pop - x1[None, :], axis=-1)

        # Continue until the generation budget is spent or the best candidate
        # lands within `tolerance` of x1 (best_fitness starts at +inf).
        def keep_going(carry):
            state, _ = carry
            return ((state.generation_counter < num_generations)
                    & (state.best_fitness >= tolerance))

        # One generation: sample candidates, shoot them, update the search.
        # tell() also increments state.generation_counter.
        def one_generation(carry):
            state, key = carry
            key, key_gen, key_eval = jax.random.split(key, 3)
            v0_pop, state = strategy.ask(key_gen, state, es_params)
            state, _ = strategy.tell(key_eval, v0_pop, endpoint_error(v0_pop),
                                     state, es_params)
            return state, key

        state = strategy.init(key, v0_init, es_params)
        state, _ = jax.lax.while_loop(keep_going, one_generation, (state, key))
        metrics = {
            "best_solution": state.best_solution,           # best v0 shot so far
            "best_fitness": state.best_fitness,             # its end-point error
            "generation_counter": state.generation_counter, # generations run
        }
        return state, metrics

    # Compiled once per call, so it always uses the current g_inv: a function
    # jitted at module level would keep whatever g_inv it was first traced
    # with if g_inv reads global variables that change later.
    return jax.jit(jax.vmap(shoot_one_pair))(x0s, x1s, keys, v0_inits)


#%% 2. Shooting method with Levenberg-Marquardt (gradient-based)
def estimate_geodesic(g_inv, x0, x1, key, *, num_restarts=10, tol=1e-3,
                      init_noise=0.5, n_steps=512):
    """
    Find the initial velocity v0 whose geodesic from x0 ends at x1, keeping
    the shortest geodesic found over several restarts.

    Each restart minimizes the end-point residual x1 - exp_map(x0, v0) over v0
    with Levenberg-Marquardt; the Jacobian comes from reverse-mode autodiff
    through the ODE solve in `euclid_geodesics.exp_map`. Restart 0 starts at
    the straight-line velocity x1 - x0; the others add Gaussian noise with
    standard deviation init_noise * |x1 - x0|, so they can converge to
    different geodesics. Among the restarts whose end-point error is below
    `tol`, the one with the shortest geodesic length
    (`euclid_geodesics.geodesic_length`) wins.

    Gradient-based version of `core/geodesics_archived.estimate_geodesic`, which takes
    the precision matrix P(x) instead of g_inv(x). One pair at a time.

    Parameters
    ----------
    g_inv : Function R^d -> R^(d,d) returning the inverse metric g^{ij}(x),
        i.e. the covariance Sigma(x).
    x0 : array, shape (d,)
        Start point.
    x1 : array, shape (d,)
        End point.
    key : jax.random.PRNGKey
        Random key for the restart perturbations.
    num_restarts : int
        Number of starting guesses.
    tol : float
        A restart counts as converged if |x1 - x(1)| < tol.
    init_noise : float
        Standard deviation of the restart perturbations, relative to |x1 - x0|.
    n_steps : int
        Number of fixed integration steps per shot, passed to `exp_map`
        (see `shoot_geodesics` for the accuracy of the default, 512).

    Returns
    -------
    best_v0 : array, shape (d,)
        Initial velocity of the shortest converged geodesic. Pass it to
        `euclid_geodesics.geodesic_path` for the path.
    """
    def resid(v0, args):
        return x1 - euclid_geodesics.exp_map(g_inv, x0, v0, n_steps)

    # Starting guesses: the straight line, then perturbed copies of it.
    noise = init_noise * jnp.linalg.norm(x1 - x0) * jax.random.normal(
        key, (num_restarts, x0.shape[0]))
    v0_inits = (x1 - x0) + noise.at[0].set(0.0)

    # Run all restarts at once. throw=False: a restart that does not converge
    # is skipped below instead of raising an error.
    def solve(v0_init):
        return optimistix.least_squares(
            resid, LevenbergMarquardt(tol * 0.1, tol * 0.1), v0_init,
            options=dict(jac="bwd"), throw=False).value
    v0s = jax.jit(jax.vmap(solve))(v0_inits)

    # Keep the shortest restart whose geodesic lands within tol of x1.
    errors = jax.vmap(lambda v: jnp.linalg.norm(resid(v, None)))(v0s)
    lengths = jax.vmap(lambda v: euclid_geodesics.geodesic_length(g_inv, x0, v))(v0s)
    lengths = jnp.where(errors < tol, lengths, jnp.inf)
    if not jnp.isfinite(lengths.min()):
        raise RuntimeError("Shooting method did not converge.")
    return v0s[jnp.argmin(lengths)]
