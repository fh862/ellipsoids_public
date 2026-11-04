"""
Geodesics between two fixed points (boundary value problem), built on the
initial value solver in `euclidization.geodesics`.

`euclidization.geodesics` integrates the geodesic equation forward from a
start point x0 and an initial velocity v0 (`exp_map`, `geodesic_path`,
`geodesic_length`). This module adds the shooting method on top of it: given
x0 and an end point x1, search for a v0 whose geodesic ends at x1. Both
solvers take a batch of pairs and differ in the optimizer:

    shoot_geodesics_cma : CMA-ES (derivative-free); returns the search state
                          and metrics.
    shoot_geodesics_lm  : Levenberg-Marquardt (gradient-based) with restarts;
                          returns the shortest converged v0 per pair.

Every function takes the inverse metric g_inv(x) -> (d, d), i.e. the Wishart
covariance Sigma(x), the same convention as `euclidization.geodesics`. For a
fitted Wishart model:

    g_inv = lambda x: model.compute_Sigmas(model.compute_U(W, x))

"""

import jax
import jax.numpy as jnp
import optimistix
from evosax.algorithms import CMA_ES
from euclidization import geodesics as euclid_geodesics
from optimistix import LevenbergMarquardt

#%% 1. Shooting method with CMA-ES (derivative-free)
def shoot_geodesics_cma(
        g_inv, x0s, x1s, keys, *,
        num_generations=20, popsize=20, tolerance=1e-3,
        v0_inits=None, n_steps=512,
    ):
    """
    For each pair, find an initial velocity v0 whose geodesic from x0 ends at
    x1, using CMA-ES (derivative-free shooting; see `shoot_geodesics_lm` for
    the gradient-based version).

    Each candidate v0 is shot with `euclid_geodesics.exp_map` and scored by
    the Euclidean distance from its end point to x1. A pair stops after
    `num_generations` or once its best score is below `tolerance`. All pairs
    run in one vmapped `jax.lax.while_loop`; finished pairs are frozen, so
    each result is independent of the batch. For one pair, pass x0[None], etc.

    The score ignores length: if several geodesics join x0 and x1, the search
    returns whichever it finds, not necessarily the shortest. The default
    start x1 - x0 favors the one nearest the straight line.


    Parameters
    ----------
    g_inv : Function R^d -> R^(d,d) returning the inverse metric g^{ij}(x),
        i.e. the covariance Sigma(x).
    x0s, x1s : array, shape (B, d)
        Start and end points.
    keys : array, shape (B, 2)
        One random key per pair, e.g. jax.random.split(key, B).
    num_generations : int
        Maximum CMA-ES generations per pair.
    popsize : int
        Candidate velocities per generation.
    tolerance : float
        Stop a pair once its best end-point error is below this (units of x).
    v0_inits : array, shape (B, d), optional
        Starting means of the searches. Defaults to x1s - x0s, the exact
        answer for a flat metric.
    n_steps : int
        Fixed integration steps per shot, passed to `exp_map`. The default,
        512, gives end-point errors of ~1e-13 on the Wishart fits; fewer steps
        are faster and still well below `tolerance` (~5e-7 at 32, 1e-8 at 64).

    Returns
    -------
    state : evosax CMA-ES states, each field with a leading axis B.
    metrics : dict, each entry with a leading axis B
        "best_solution": best v0 found, (B, d);
        "best_fitness": its end-point error, (B,);
        "generation_counter": generations run, (B,).
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
def shoot_geodesics_lm(g_inv, x0s, x1s, keys, *, num_restarts=10, tol=1e-3,
                       init_noise=0.5, n_steps=512):
    """
    For each pair, find the initial velocity v0 whose geodesic from x0 ends at
    x1, using Levenberg-Marquardt (gradient-based shooting; see
    `shoot_geodesics_cma` for the derivative-free version). Returns only the
    velocities.

    Each restart minimizes x1 - exp_map(x0, v0), with the Jacobian from
    reverse-mode autodiff through the ODE solve. Restart 0 starts at x1 - x0;
    the others add Gaussian noise with SD init_noise * |x1 - x0|. Among the
    restarts that land within `tol` of x1, the shortest geodesic is kept.

    All pairs and restarts run in one vmapped call; run time grows with
    B * num_restarts. On the sub10 suprathreshold pairs every restart found
    the same geodesic, so fewer restarts suffice there. For one pair, pass
    x0[None], etc.

    Parameters
    ----------
    g_inv : Function R^d -> R^(d,d) returning the inverse metric g^{ij}(x),
        i.e. the covariance Sigma(x).
    x0s : array, shape (B, d)
        Start points.
    x1s : array, shape (B, d)
        End points.
    keys : array, shape (B, 2)
        One random key per pair for the restart perturbations, e.g.
        jax.random.split(key, B).
    num_restarts : int
        Number of starting guesses per pair.
    tol : float
        A restart counts as converged if |x1 - x(1)| < tol.
    init_noise : float
        Standard deviation of the restart perturbations, relative to |x1 - x0|.
    n_steps : int
        Number of fixed integration steps per shot, passed to `exp_map`
        (see `shoot_geodesics_cma` for the accuracy of the default, 512).

    Returns
    -------
    best_v0s : array, shape (B, d)
        Initial velocity of the shortest converged geodesic for each pair.
        Pass it to `euclid_geodesics.geodesic_path` for the path.

    """
    def solve_one_pair(x0, x1, key):
        def resid(v0, args):
            return x1 - euclid_geodesics.exp_map(g_inv, x0, v0, n_steps)

        # Starting guesses: the straight line, then perturbed copies of it.
        noise = init_noise * jnp.linalg.norm(x1 - x0) * jax.random.normal(
            key, (num_restarts, x0.shape[0]))
        v0_inits = (x1 - x0) + noise.at[0].set(0.0)

        # Run all restarts at once. throw=False: a restart that does not
        # converge is skipped below instead of raising an error.
        def solve(v0_init):
            return optimistix.least_squares(
                resid, LevenbergMarquardt(tol * 0.1, tol * 0.1), v0_init,
                options=dict(jac="bwd"), throw=False).value
        v0s = jax.vmap(solve)(v0_inits)

        # Keep the shortest restart whose geodesic lands within tol of x1.
        errors = jax.vmap(lambda v: jnp.linalg.norm(resid(v, None)))(v0s)
        lengths = jax.vmap(lambda v: euclid_geodesics.geodesic_length(g_inv, x0, v))(v0s)
        lengths = jnp.where(errors < tol, lengths, jnp.inf)
        best = jnp.argmin(lengths)
        return v0s[best], jnp.isfinite(lengths[best])

    # Compiled once per call, so it always uses the current g_inv (see
    # `shoot_geodesics_cma`).
    best_v0s, converged = jax.jit(jax.vmap(solve_one_pair))(x0s, x1s, keys)
    if not bool(jnp.all(converged)):
        failed = jnp.flatnonzero(~converged).tolist()
        raise RuntimeError(f"Shooting method did not converge for pairs {failed}.")
    return best_v0s
