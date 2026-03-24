import jax
import jax.numpy as jnp
from . import wishart_process


def estimate_loglikelihood(
        W, model, data, key, num_samples, bandwidth, simulation_func
    ):
    """
    Parameters
    ----------
    W : array
        Array holding the coefficients of the basis function.
        These are the only parameters we want to optimize or
        run Bayesian inference on.

    model : WishartProcessModel
        Model instance, used to compute ellipsoids from `W`.

    data : Tuple containing
        y : array
            Array holding subject responses, y.shape = (num_trials,)
            Each trial is a binary response in coded in {0, 1}.
            A value of zero means that the subject thinks x0 is
            closer to xref on that trial. A value of one means that
            the subject thinks x1 is closer to xref on that trial.
        mref : array
            Array holding the reference stimulus on each trial.
            mref.shape = (num_trials, num_dims).
        mprobe : array
            Array holding probe stimulus on each trial.
            mprobe.shape = (num_trials, num_dims).

    key : jax.random.PRNGKey
        Key for the random number generator

    num_samples : int
        Number of samples to use for the Monte Carlo
        estimate of the likelihood.

    bandwidth : float
        Smoothing parameter for the Monte Carlo estimate
        of the likelihood. Note that as bandwidth goes to
        zero the function will not be differentiable with
        respect to `W` (which is bad!).
    
    simulation_func: function
        Use 'simulate_oddity_reference' for oddity tasks where the reference 
            is fixed.
        Use 'simulate_oddity' when the reference and the other two probe 
            stimuli are shuffled.

    Returns
    -------
    log_likelihood : float
        Log likelihood of W given data (y, xref, x0, and x1).
        Averaged across trials.
    """
    
    # Unpack data.
    y, mref, mprobe = data

    # Compute (Uref, Uprobe) to specify ellipsoids.
    Uref = model.compute_U(W, mref)
    Uprobe = model.compute_U(W, mprobe)

    # Run simulation to evaluate predicted responses.
    prob_correct = oddity_prediction(
        (mref, mprobe, Uref, Uprobe),
        jax.random.split(key, num=len(y)),
        num_samples, bandwidth, model.diag_term,
        simulation_func
    )
    prob_correct = jnp.clip(prob_correct, 0.001, 0.999)

    # Evaluate log likelihood of predictions given
    # true responses y.
    return jnp.mean(
        y * jnp.log(prob_correct) + (1 - y) * jnp.log(1-prob_correct)
    )


def oddity_prediction(
        params, key, num_samples, bandwidth, diag_term, simulation_func
    ):
    """
    Parameters
    ----------
    params : Tuple
        Holds means and covariances for the reference and probe stimuli.

    key : jax.random.PRNGKey
        Seed for the random number generator.
    
    num_samples : int
        How many samples to use to approximate the probability
    
    bandwidth : float
        Smoothing parameter. Warning - as this goes to zero the function becomes
        nondifferentiable with respect to `params`.
        
    simulation_func: function
        Use 'simulate_oddity_reference' for oddity tasks where the reference 
            is fixed.
        Use 'simulate_oddity' when the reference and the other two probe 
            stimuli are shuffled.
    
    Returns
    -------
    prob_correct : float
        The probability of the subject making a correct response.
    """
    
    # Simulate outcomes with the current choice of parameters. Outcomes are
    # are negative when the subject chooses probe stimulus as the odd one out
    outcomes = simulation_func(params, key, num_samples, diag_term)

    # Evaluate the cumulative density function at zero, which gives
    # the probability of making a correct choice. 
    return approx_cdf(0.0, outcomes, bandwidth)

def simulate_oddity_one_trial(params, key, num_samples, diag_term):
    """
    Simulate oddity task outcomes.

    Returns
    -------
    result : float
        A positive number indicates an incorrect trial, dist(z0, z1) 
        is larger than at least one of dist(z0, z2) and dist(z1, z2).
        A negative number indicates a correct trial, dist(z0, z1)
        is the smallest distance, meaning that z2 is correctly 
        identified as the odd-one-out.
    """

    # Unpack parameters
    #     mref, mprobe == mean parameters for reference and probe stimuli
    #     Uref, Uprobe == Specifies covariance for reference and probe stimuli
    mref, mprobe, Uref, Uprobe = params

    # Number of stimulus dimensions
    ndims_cov, ndims_extra = Uref.shape

    # Generate random draws from isotropic, standard gaussians
    keys = jax.random.split(key, num=6)
    nn0 = jax.random.normal(keys[0], shape=(num_samples, ndims_extra))
    nn1 = jax.random.normal(keys[1], shape=(num_samples, ndims_extra))
    nn2 = jax.random.normal(keys[2], shape=(num_samples, ndims_extra))
    _nn0 = jax.random.normal(keys[3], shape=(num_samples, ndims_cov))
    _nn1 = jax.random.normal(keys[4], shape=(num_samples, ndims_cov))
    _nn2 = jax.random.normal(keys[5], shape=(num_samples, ndims_cov))

    # Re-scale and translate the noisy samples to have the correct mean and
    # covariance.
    #     z0 ~ Normal(mref, Uref @ Uref.T).
    #     z1 ~ Normal(mref, Uref @ Uref.T).
    #     z2 ~ Normal(mprobe, Uprobe @ Uprobe.T).
    z0 = nn0 @ Uref.T + mref[None, :ndims_cov] + (_nn0 * jnp.sqrt(diag_term))
    z1 = nn1 @ Uref.T + mref[None, :ndims_cov] + (_nn1 * jnp.sqrt(diag_term))
    z2 = nn2 @ Uprobe.T + mprobe[None, :ndims_cov] + (_nn2 * jnp.sqrt(diag_term))

    # Compute covariances
    S0 = Uref @ Uref.T + jnp.diag(jnp.full(ndims_cov, diag_term))
    S1 = Uprobe @ Uprobe.T + jnp.diag(jnp.full(ndims_cov, diag_term))

    # Average covariances
    Sbar = (2 / 3) * S0 + (1 / 3) * S1

    # Compute squared Mahalanobis distances
    r01 = z0 - z1
    r02 = z0 - z2
    r12 = z1 - z2

    z0_to_z1 = jnp.sum(r01 * jnp.linalg.solve(Sbar, r01.T).T, axis=1)
    z0_to_z2 = jnp.sum(r02 * jnp.linalg.solve(Sbar, r02.T).T, axis=1)
    z1_to_z2 = jnp.sum(r12 * jnp.linalg.solve(Sbar, r12.T).T, axis=1)

    # Return signed difference.
    return z0_to_z1 - jnp.minimum(z0_to_z2, z1_to_z2)

def simulate_oddity_suprathres_one_trial(params, key, num_samples, diag_term):
    """
    Simulate oddity task for the supra-threshold expt
    """

    # Unpack parameters
    #     mref, mprobe == mean parameters for reference and probe stimuli
    #     Uref, Uprobe == Specifies covariance for reference and probe stimuli
    mref, mprobe1, mprobe2, Uref, Uprobe1, Uprobe2 = params

    # Number of stimulus dimensions
    ndims_cov, ndims_extra = Uref.shape

    # Generate random draws from isotropic, standard gaussians
    keys = jax.random.split(key, num=6)
    nn0 = jax.random.normal(keys[0], shape=(num_samples, ndims_extra))
    nn1 = jax.random.normal(keys[1], shape=(num_samples, ndims_extra))
    nn2 = jax.random.normal(keys[2], shape=(num_samples, ndims_extra))
    _nn0 = jax.random.normal(keys[3], shape=(num_samples, ndims_cov))
    _nn1 = jax.random.normal(keys[4], shape=(num_samples, ndims_cov))
    _nn2 = jax.random.normal(keys[5], shape=(num_samples, ndims_cov))

    # Re-scale and translate the noisy samples to have the correct mean and
    # covariance.
    #     z0 ~ Normal(mref, Uref @ Uref.T).
    #     z1 ~ Normal(mref, Uref @ Uref.T).
    #     z2 ~ Normal(mprobe, Uprobe @ Uprobe.T).
    z0 = nn0 @ Uref.T + mref[None, :] + (_nn0 * jnp.sqrt(diag_term))
    z1 = nn1 @ Uprobe1.T + mprobe1[None, :] + (_nn1 * jnp.sqrt(diag_term))
    z2 = nn2 @ Uprobe2.T + mprobe2[None, :] + (_nn2 * jnp.sqrt(diag_term))

    # Compute covariances
    S0 = Uref @ Uref.T + jnp.diag(jnp.full(ndims_cov, diag_term))
    S1 = Uprobe1 @ Uprobe1.T + jnp.diag(jnp.full(ndims_cov, diag_term))
    S2 = Uprobe2 @ Uprobe2.T + jnp.diag(jnp.full(ndims_cov, diag_term))

    # Average covariances
    Sbar = (S0 + S1 + S2)/3

    # Compute residuals
    r1 = z1 - z0
    r2 = z2 - z0

    # Compute squared Mahalanobis distances
    z1_to_z0 = jnp.sum(r1 * jnp.linalg.solve(Sbar, r1.T).T, axis=1)
    z2_to_z0 = jnp.sum(r2 * jnp.linalg.solve(Sbar, r2.T).T, axis=1)

    # Return signed difference.
    return z1_to_z0 - z2_to_z0


def simulate_oddity_one_trial_reference(params, key, num_samples, diag_term):
    """
    Simulate oddity task outcomes. 
    Note: This function differs from simulate_oddity_one_trial in that it 
        assumes the location of the reference (mref) is fixed and known to the 
        participants. The task is to determine between m0 (which is identical 
        to mref) and m1, which one is more dissimilar to mref.

    Returns
    -------
    result : float
        Squared distance of (z0 to zref) minus squared
        distance of (z1 to zref). A positive number
        indicates that the subject believes m1 is closer
        to mref than m0. A negative number indicates that
        m0 is closer to mref than m0.
    """

    # Unpack parameters
    #     mref, m0, m1 == mean parameters for reference and probe stimuli
    #     Uref, U0, U1 == Specifies covariance for reference and probe stimuli
    mref, m1, Uref, U1 = params

    # Number of stimulus dimensions
    ndims_cov, ndims_extra = Uref.shape

    # Generate random draws from isotropic, standard gaussians
    keys = jax.random.split(key, num=6)
    nnref = jax.random.normal(keys[0], shape=(num_samples, ndims_extra))
    nn0 = jax.random.normal(keys[1], shape=(num_samples, ndims_extra))
    nn1 = jax.random.normal(keys[2], shape=(num_samples, ndims_extra))
    _nnref = jax.random.normal(keys[3], shape=(num_samples, ndims_cov))
    _nn0 = jax.random.normal(keys[4], shape=(num_samples, ndims_cov))
    _nn1 = jax.random.normal(keys[5], shape=(num_samples, ndims_cov))

    # Re-scale and translate the noisy samples to have the correct mean and
    # covariance. For example, zref ~ Normal(mref, Uref @ Uref.T).
    zref = nnref @ Uref.T + mref[None, :] + (_nnref * jnp.sqrt(diag_term))
    z0 = nn0 @ Uref.T + mref[None, :] + (_nn0 * jnp.sqrt(diag_term))
    z1 = nn1 @ U1.T + m1[None, :] + (_nn1 * jnp.sqrt(diag_term))

    # Compute covariances
    Sref = Uref @ Uref.T + jnp.diag(jnp.full(ndims_cov, diag_term))
    S0 = Uref @ Uref.T + jnp.diag(jnp.full(ndims_cov, diag_term))
    S1 = U1 @ U1.T + jnp.diag(jnp.full(ndims_cov, diag_term))

    # Average covariances
    Sbar = (Sref + S0 + S1) / 3

    # Compute residuals
    r0 = z0 - zref
    r1 = z1 - zref

    # Compute squared Mahalanobis distances
    z0_to_zref = jnp.sum(r0 * jnp.linalg.solve(Sbar, r0.T).T, axis=1)
    z1_to_zref = jnp.sum(r1 * jnp.linalg.solve(Sbar, r1.T).T, axis=1)

    # Return signed difference.
    return z0_to_zref - z1_to_zref

def approx_cdf_one_trial(x, xs, h):
    """
    Approximate the cumulative density function of a distribution at `x` given
    sampled values `xs = [x1, ..., xn]`. The density function is approximated
    using kernel smoothing with a logistic density (hyperbolic secant) kernel.
    The bandwidth parameter `h` controls the width of this smoothing kernel.

    Parameters
    ----------
    x : float
        Value to evaluate the cumulative density function.
    
    xs : array
        Sampled values from the underlying probability distribution.
    
    h : float
        Bandwidth parameter (larger values = more smoothing). As h goes to zero
        we get the cdf of the empirical distribution (which is not
        differentiable).

    Returns
    -------
    cumulative_density : float
        Estimate of the cumulative density
    """
    return jnp.mean(jax.lax.logistic((x - xs) / h))


# ===========
# Use jax.vmap to make a function that simulates many trials.
# ===========
simulate_oddity = jax.vmap(
    simulate_oddity_one_trial, ((0, 0, 0, 0), 0, None, None), 0
)
simulate_oddity_suprathres = jax.vmap(
    simulate_oddity_suprathres_one_trial, ((0, 0, 0, 0, 0, 0), 0, None, None), 0
)
simulate_oddity_reference = jax.vmap(
    simulate_oddity_one_trial_reference, ((0, 0, 0, 0), 0, None, None), 0
)
approx_cdf = jax.vmap(
    approx_cdf_one_trial, (None, 0, None), 0
)
# ===========
