"""Probability functions for the random effects model."""

import jax.numpy as jnp
import distrax

from modularbayes._src.typing import Any, Array, Batch, Dict, Optional, SmiEta


# Joint distribution (data and params)
def log_prob_joint(
    batch: Batch,
    posterior_sample_dict: Dict[str, Any],
    smi_eta: Optional[SmiEta] = None,
) -> Array:
  """Compute the joint probability for the HPV model.

  The joint log probability of the model is given by

  .. math::

    log_prob(Y,tau,beta,sigma) &= \prod_{i=1}^{N} log_prob(Y_i \| \beta, \sigma) \\
                            &+ log_prob(beta \| \tau)  \\
                            &+ log_prob(\tau \| \sigma)  \\
                            &+ log_prob(\sigma).

  Optionally, if `smi_eta` is provided, the link between groups is reduced

  .. math::

    log_prob(Y,tau,beta,sigma) &= \prod_{i=1}^{N} log_prob(Y_i \| \beta, \sigma) \\
                            &+ log_prob(beta \| \tau)  \\
                            &+ \eta * log_prob(\tau \| \sigma)  \\
                            &+ log_prob(\sigma).

  Args:
    batch: Dictionary with the data. Must contains 2 items 'Y' and 'group',
      each one of shape (n,).
    posterior_sample_dict: Dictionary with values for the model parameters. Must
      contain 3 items: 'tau', 'beta' and 'sigma', arrays with shapes (s, 1),
      (s, num_groups) and (s, num_groups), respectively.
    smi_eta: Optional dictionary with the power to be applied on each module of
      the likelihood. Must contain 'groups', an array of shape (num_groups,).

  Output:
    Array of shape (s,) with log joint probability evaluated on each value of
    the model parameters.
  """
  num_samples, _ = posterior_sample_dict['sigma'].shape

  # Compute pointwise log-likelihood
  loglik_pointwise = distrax.Normal(
      loc=posterior_sample_dict['beta'][:, batch['group']],
      scale=posterior_sample_dict['sigma'][:,
                                           batch['group']]).log_prob(batch['Y'])

  # Add over observations
  loglik = loglik_pointwise.sum(axis=-1)

  assert loglik.shape == (num_samples,)

  # Define priors
  def log_prob_beta(
      beta: Array,
      tau: Array,
      smi_eta_groups: Optional[Array] = None,
  ):
    num_samples, num_groups = beta.shape

    beta_prior_scale = jnp.broadcast_to(tau, (num_samples, num_groups))
    if smi_eta_groups is not None:
      beta_prior_scale = jnp.broadcast_to(
          tau / jnp.expand_dims(smi_eta_groups, axis=0),
          (num_samples, num_groups))
    else:
      beta_prior_scale = jnp.broadcast_to(tau, (num_samples, num_groups))

    log_prob = distrax.Normal(
        loc=jnp.zeros((num_samples, num_groups)),
        scale=beta_prior_scale,
    ).log_prob(beta)
    # Add over groups
    log_prob = log_prob.sum(axis=-1)

    return log_prob

  # If smi, multiply by eta

  def log_prob_tau(
      tau: Array,
      sigma: Array,
      num_obs_groups: Array,
  ):
    tau = tau.squeeze(-1)

    log_prob = -jnp.log(tau**2 + (sigma**2 / num_obs_groups).mean(axis=-1))

    return log_prob

  def log_prob_sigma(sigma: Array):
    return (-2 * jnp.log(sigma)).sum(axis=-1)

  # Everything together
  log_prob = (
      loglik + log_prob_beta(
          beta=posterior_sample_dict['beta'],
          tau=posterior_sample_dict['tau'],
          smi_eta_groups=smi_eta['groups'] if smi_eta else None,
      ) + log_prob_tau(
          tau=posterior_sample_dict['tau'],
          sigma=posterior_sample_dict['sigma'],
          num_obs_groups=batch['num_obs_groups'],
      ) + log_prob_sigma(posterior_sample_dict['sigma']))

  assert log_prob.shape == (num_samples,)

  return log_prob
