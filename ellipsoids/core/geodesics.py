import jax
import jax.numpy as jnp
import numpy as np
import diffrax
from diffrax import ODETerm, Dopri5, SaveAt, PIDController 
import optimistix
from optimistix import LevenbergMarquardt
from tqdm import tqdm, trange
from evosax.algorithms import CMA_ES

# Constants describing ode solver and step size.
ODESOLVER = Dopri5() 
TS = jnp.linspace(0.0, 1.0, 1001)
DT = 1e-3
def geodesic_vector_field(P):
    """
    Given a function `P(x)` that returns the inverse
    covariance (a.k.a. precision matrix) at location
    `x` in stimulus space, define a function that
    computes the vector field of geodesic flows:
    `dxdt` and `dvdt` are respectively the velocity
    and acceleration vectors along the geodesic path.
    """
    jacP = jax.jacobian(P)
    def vector_field(t, state, args):
        x, v = state
        Pdx = jacP(x)
        q1 = 0.5 * jnp.einsum("jki,j,k->i",Pdx, v, v)
        q2 = jnp.einsum("ilp,l,p->i", Pdx, v, v)
        dxdt = v
        dvdt = jnp.linalg.solve(P(x), q1 - q2)
        return (dxdt, dvdt)
    return vector_field

def shooting_geodesic(x0, v0, precision_field):
    """
    Computes the full geodesic path starting from initial
    position `x0` and initial velocity `v0`. 

    Returns
    ----
    path: ndarray (num_timesteps x num_dims)
    time_axis: ndarray (num_timesteps)
    """
    sol = diffrax.diffeqsolve(
        ODETerm(geodesic_vector_field(precision_field)),
        ODESOLVER, t0=0, t1=1, dt0=DT, y0=(x0, v0),
        saveat=SaveAt(t0=True, t1=True, steps=True)
        #stepsize_controller=PIDController(rtol=1e-10, atol=1e-10), # max_steps=100000#, 
    )
    idx = ~jnp.isinf(sol.ts)  # diffrax sometimes outputs extra timesteps
    return sol.ys[0][idx], sol.ts[idx]

def exponential_map(x0, v0, odeterm, odesolver, dt0=0.001):
    """
    Compute the geodesic starting at position `x0` and with
    initial velocity `v0`. Uses package diffrax to numerically
    solve the ODE.
    """
    return diffrax.diffeqsolve(
        odeterm, odesolver, t0=0, t1=1, dt0=dt0, y0=(x0, v0),
        saveat=SaveAt(t0=True, t1=True, steps=True)
    ).ys[0]

def estimate_geodesic(P, x0, x1, key, dt0=0.1, num_restarts=10, tol=1e-3):
    term = ODETerm(geodesic_vector_field(P))
    ode_solver = Dopri5()
    optimizer = LevenbergMarquardt(tol * 0.1, tol * 0.1)
    resids = jax.jit(
        lambda v0, args: (x1 - exponential_map(x0, v0, term, ode_solver, dt0=dt0)).ravel()
    )
    z = (x1 - x0) / jnp.linalg.norm(x1 - x0)
    min_speed = jnp.inf
    best_v0 = None
    for k in tqdm(jax.random.split(key, num_restarts)):
        v0_init = (x1 - x0) + (
            0.0 * jnp.linalg.norm(x1 - x0) * jax.random.normal(k, shape=x0.shape)
        )
        result = optimistix.least_squares(
            resids, optimizer, v0_init, options=dict(jac="bwd")
        )
        converged = jnp.linalg.norm(result.state.f_info.residual) < tol
        speed = jnp.linalg.norm(result.value)
        if converged and (speed < min_speed):
            min_speed = speed
            best_v0 = result.value
    if best_v0 is None:
        raise RuntimeError("Shooting method did not converge.")
    def geodesic_simulator(dt0, saveat):
        return diffrax.diffeqsolve(
            term, ode_solver, t0=0, t1=1, dt0=dt0, y0=(x0, best_v0),
            saveat=saveat
        ).ys[0]
    return best_v0, geodesic_simulator

def estimate_v0(
        x0, x1, precision_field, key, 
        num_generations=20, popsize=20, tolerance=1e-3, solution_dim = 2,
    ):
    """
    Attempts to find a good geodesic path that starts at
    position `x0` and ends at position `x1`.

    We do this by using an evolutionary strategy (CMA-ES) to
    find a good initial velocity `v0` and using it to compute
    a shooting geodesic that ends close to `x1`.

    Returns
    ----
    path: ndarray (num_timesteps x num_dims)
    time_axis: ndarray (num_timesteps)
    """
    # Initialize evolutionary strategy
    # strategy = CMA_ES(popsize=popsize)
    # es_params = strategy.default_params
    # es_params = es_params.replace(sigma_init=jnp.array([0.25, 0.25]))
    # state = strategy.initialize(key, es_params)
    # state = state.replace(best_fitness=jnp.finfo(jnp.float64).max)
    
    # Define dimensionality of v0
    # Step 1: Setup
    #key = jax.random.PRNGKey(0)
    dummy_solution = jnp.array([0.25] * solution_dim)
    
    # Step 2: Initialize strategy with required positional args
    strategy = CMA_ES(population_size=popsize, solution=dummy_solution)
    print(strategy.init)
    
    # Step 3: Create customized parameters
    es_params = strategy.default_params
    
    # Step 4: Initialize state
    state = strategy.init(key, dummy_solution, es_params)

    # Vmap the exponential map function over initial velocities.
    emap = jax.vmap(shooting_geodesic, in_axes=(None, 0, None))
    
    # Run evolutionary strategy
    for t in trange(num_generations):
        key, key_gen, key_eval = jax.random.split(key, 3)
        v0, state = strategy.ask(key_gen, state, es_params)
        x1e = emap(x0, v0, precision_field)[0][:, -1, :]
        fitness = np.sqrt(np.sum((x1[None, :] - x1e) ** 2, axis=-1))
        state, metrics = strategy.tell(key_eval, v0, fitness, state, es_params)
        if metrics["best_fitness"] < tolerance:
            break

    return state, metrics    
            
def estimate_path_cost(xt, ts, precision_field):
    """
    Given a geodesic path `xt` sampled over a grid of time points
    specified in `ts`, compute the path cost (i.e. geodesic distance)
    """
    dts = jnp.diff(ts)
    v = jnp.diff(xt, axis=0) / dts[:, None]
    x = 0.5 * (xt[1:] + xt[:-1])
    costs = jnp.einsum("tij,ti,tj->t", jax.vmap(precision_field)(x), v, v)
    return jnp.sqrt(jnp.sum(costs * dts))

def segment_costs(xt, ts, precision_field):
    xt = jnp.asarray(xt)
    ts = jnp.asarray(ts)

    dts = jnp.diff(ts)                                 # (T-1,)
    v = jnp.diff(xt, axis=0) / dts[:, None]            # (T-1, D)
    x_mid = 0.5 * (xt[1:] + xt[:-1])                   # (T-1, D)
    P_mid = jax.vmap(precision_field)(x_mid)           # (T-1, D, D)

    quad = jnp.einsum('tij,ti,tj->t', P_mid, v, v)     # v^T P v per segment
    seg_lengths = jnp.sqrt(quad) * dts                 # length contribution per segment
    return jnp.array(seg_lengths)

#%%
# ---- builders that bind P ----
def make_shooting_geodesic_fixed(P):
    term = ODETerm(geodesic_vector_field(P))
    @jax.jit
    def shooting(x0, v0):
        sol = diffrax.diffeqsolve(
            term, ODESOLVER,
            t0=0.0, t1=1.0, dt0=DT, y0=(x0, v0),
            saveat=SaveAt(ts=TS)
        )
        return sol.ys[0]  # (T, D)
    return shooting

def make_shoot_pop_end(P):
    shooting = make_shooting_geodesic_fixed(P)
    return jax.jit(jax.vmap(lambda v0, x0: shooting(x0, v0)[-1], in_axes=(0, None)))

# ---- dataset-specific P (binds W_init etc.) ----
def load_modelW(m, W_in):
    global model, W
    model, W = m, W_in

@jax.jit
def P(x):
    return jnp.linalg.inv(
        model.compute_Sigmas(model.compute_U(W, x))
    )

# compile helpers for this dataset
shooting_geodesic_fixed = make_shooting_geodesic_fixed(P)
shoot_pop_end           = make_shoot_pop_end(P)

# ---- CMA-ES per-pair  ----
def run_cma_one(x0, x1, key, popsize=64, num_generations=30):
    D = x0.shape[-1]
    strategy = CMA_ES(population_size=popsize, solution=jnp.zeros((D,)))
    params = strategy.default_params
    state = strategy.init(key, jnp.zeros((D,)), params)

    def one_gen(state, rng):
        key_gen, key_eval = jax.random.split(rng, 2)
        v0_pop, state = strategy.ask(key_gen, state, params)          # (pop, D)
        x1_pred = shoot_pop_end(v0_pop, x0)                           # (pop, D)
        fitness = jnp.linalg.norm(x1_pred - x1[None, :], axis=-1)     # (pop,)
        state, metrics = strategy.tell(key_eval, v0_pop, fitness, state, params)
        return state, metrics

    keys = jax.random.split(key, num_generations)
    state, metrics = jax.lax.scan(one_gen, state, keys)

    # pick best v0 from metrics (evosax version-dependent)
    best_v0 = getattr(state, "mean", getattr(state, "param_mean", None))

    return best_v0, state, metrics

run_cma_one_jit = jax.jit(run_cma_one, static_argnames=("popsize","num_generations"))

def batch_estimate_v0(starts, ends, keys, popsize=64, num_generations=30):
    def _one(x0, x1, key):
        return run_cma_one_jit(x0, x1, key, popsize, num_generations)[0]
    return jax.vmap(_one, in_axes=(0, 0, 0))(starts, ends, keys)

def batch_paths_and_dists(starts, v0s):
    # uses the compiled shooting_geodesic_fixed bound to P
    paths = jax.vmap(shooting_geodesic_fixed)(starts, v0s)  # (B, T, D)
    dts = jnp.diff(TS)
    def one_dist(path):
        v = jnp.diff(path, axis=0) / dts[:, None]
        x_mid = 0.5 * (path[1:] + path[:-1])
        P_mid = jax.vmap(P)(x_mid)
        integrand = jnp.einsum("tij,ti,tj->t", P_mid, v, v)
        return jnp.sqrt(jnp.sum(integrand * dts))
    dists = jax.vmap(one_dist)(paths)
    return paths, dists