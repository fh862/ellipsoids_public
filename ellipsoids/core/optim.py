from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from . import wishart_process, oddity_task

# ==================
# Objective function
# ==================

@partial(jax.jit, static_argnums=(1, 4, 6))
def objective(
        W, model, data, key, mc_samples, bandwidth, simulation_func,
    ):
    
    # Unpack data.
    y, mref, mprobe = data

    # Scale the log posterior by the number of samples
    # so that the learning rate is easier to tune.
    return -1 * (
        oddity_task.estimate_loglikelihood(
            W, model, data, key, mc_samples, bandwidth, simulation_func
        ) + (model.logprior_density_W(W) / len(y))
    )

obj_and_grad = jax.value_and_grad(objective)

@partial(jax.jit, static_argnums=(1, 4, 6))
def objective_no_prior(
        W, model, data, key, mc_samples, bandwidth, simulation_func,
    ):
    
    # Unpack data.
    y, mref, mprobe = data

    # Scale the log posterior by the number of samples
    # so that the learning rate is easier to tune.
    return -1 * (
        oddity_task.estimate_loglikelihood(
            W, model, data, key, mc_samples, bandwidth, simulation_func
        )
    )

obj_and_grad_no_prior = jax.value_and_grad(objective_no_prior)

# =============

def initialize_model(
        data, key, num_sims=1000,
        degree=5, num_dims=3, extra_dims=1,
        mc_samples=1000, bandwidth=1e-6
    ):

    k1, k2, k3, key = jax.random.split(key, 4)
    decay_rates = jax.random.uniform(k1, shape=(num_sims,))
    cov_scales = jax.random.uniform(k2, minval=-4, maxval=0, shape=(num_sims,)) ** 10
    diag_terms = jax.random.uniform(k3, minval=-8, maxval=-4, shape=(num_sims,)) ** 10
    models, vals = [], []

    for i in trange(num_sims):
        model = wishart_process.WishartProcessModel(
            degree,      # Degree of the polynomial basis functions
            num_dims,    # Number of stimulus dimensions
            extra_dims,  # Number of extra inner dimensions in `U`.
            cov_scales[i],   # Scale parameter for prior on `W`.
            decay_rates[i],  # Geometric decay rate on `W`.
            diag_terms[i],   # Diagonal term setting minimum variance for the ellipsoids.
        )
        prior_key, loglike_key, key = jax.random.split(key, 3)
        vals.append(
            jnp.mean(jnp.array(
                [oddity_task.estimate_loglikelihood(
                    model.sample_W_prior(k),
                    model, data, loglike_key, mc_samples, bandwidth
                ) for k in jax.random.split(prior_key, 10)]
            ))
        )
        models.append(model)

    return models, jnp.array(vals)


def optimize_posterior(
        W, data, model, key,
        opt_params, 
        simulation_func,
        total_steps=1000,
        save_every=10,
        show_progress=True,
        mask = None,
        use_prior = True
    ):
    """
    Use stochastic gradient descent with momentum to maximize
    the log posterior probability of `W`.
    """

    # Define and initialize optimization method.
    optimizer = optax.sgd(
        learning_rate=opt_params["learning_rate"],
        momentum=opt_params["momentum"]
    )
    opt_state = optimizer.init(W)

    iters = jnp.arange(0, total_steps, save_every)
    objhist = np.zeros(iters.size)

    pbar = trange(total_steps) if show_progress else range(total_steps)

    min_val = jnp.inf
    safe_W = jnp.copy(W)

    for t in pbar:
        key, _ = jax.random.split(key)
        
        if use_prior:
            val, grad = obj_and_grad(
                W, model, data, key,
                opt_params["mc_samples"],
                opt_params["bandwidth"],
                simulation_func,
            )
        else:
            val, grad = obj_and_grad_no_prior(
                W, model, data, key,
                opt_params["mc_samples"],
                opt_params["bandwidth"],
                simulation_func,
            )

        if (t % save_every) == 0:
            objhist[t // save_every] = val

        if jnp.isnan(val) or (val > (1e2 * min_val)):
            W = jnp.copy(safe_W)
            opt_params["learning_rate"] = 0.5 * opt_params["learning_rate"]
            optimizer = optax.sgd(
                learning_rate=opt_params["learning_rate"],
                momentum=opt_params["momentum"]
            )
            opt_state = optimizer.init(W)

        else:            
            updates, opt_state = optimizer.update(grad, opt_state)
            
            # Apply the mask to the updates
            if mask is not None:
                updates = jnp.where(mask, updates, 0)
            W = optax.apply_updates(W, updates)

            # Apply the mask to the parameters as well (to ensure they stay at 0)
            if mask is not None:
                W = jnp.where(mask, W, 0)
            if val < min_val:
                safe_W = jnp.copy(W)
                min_val = val

    return W, jnp.array(iters), jnp.array(objhist)

