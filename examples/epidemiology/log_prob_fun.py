"""Probability functions for the Epidemiology model."""

import jax
import jax.numpy as jnp
import distrax
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from modularbayes._src.typing import (Any, Array, Batch, Dict, List, Optional,
                                 SmiEta)


# Pointwise log-likelihood
def log_lik(Z: int, Y: int, N: int, T: int, phi: float, theta: Array) -> float:
  """Observational model for the Epidemiology data.

  The observational model is given by:
  :math:`
    Z ~ Binomial( N, \phi )
    Y ~ Poisson( T/1000 * \exp( \theta_1 + \theta_2 * \phi ) )
  `

  Args:
    Z: Number of women infected with high-risk HPV in a sample of size N.
    Y: Number of cervical cancer cases arising from T woman-years of follow-up.
    N: Number of participants in the HPV prevalence survey.
    T: Population size covered by the cancer registry corresponding to the area
      of the survey.
    phi: High-risk HPV prevalence rate.
    theta: Intercept and slope for the log linear relationship between high-risk
      HPV prevalence and cancer incidence in the same population.

  Returns:
    Array (lenght 2) with the log-likehood for the Binomial and Poisson modules.
  """

  ## Disease model
  log_incidence = theta[0] + theta[1] * phi
  mu = T * (1. / 1000) * jnp.exp(log_incidence)

  log_prob_z_given_phi = tfd.Binomial(total_count=N, probs=phi).log_prob
  log_prob_y_given_phi_theta = tfd.Poisson(rate=mu).log_prob

  log_prob_z_y = jnp.stack(
      [log_prob_z_given_phi(Z),
       log_prob_y_given_phi_theta(Y)], axis=-1)

  return log_prob_z_y


def log_lik_vectorised(
    Z: Array,
    Y: Array,
    N: Array,
    T: Array,
    phi: Array,
    theta: Array,
) -> List[Array]:
  """Compute vectorized log-likelihood function in the epidemiology model.

  Compute the log-likelihood across data samples and parameter samples.

  Args:
    Z: Array of shape (n_z,). Number of women infected with high-risk HPV in a
      sample of size N.
    Y: Array of shape (n_y,). Number of cervical cancer cases arising from T
      woman-years of follow-up.
    N: Array of shape (n_z,). Number of participants in the HPV prevalence
      survey.
    T: Array of shape (n_y,). Population size covered by the cancer registry
      corresponding to the area of the survey.
    phi: Array of shape (s, n_z). High-risk HPV prevalence rate.
    theta: Array of shape (s, 2). Intercept and slope for the log linear
      relationship between high-risk HPV prevalence and cancer incidence in the
      same population.

  Output:
    List with two arrays, of shapes (s, n_z) and (s, n_y) with each entry
    corresponding to the pointwise log-likelihood of the model.
  """

  # Vectorise along n data observations
  log_lik_vector_fun0 = jax.vmap(log_lik, in_axes=[0, 0, 0, 0, 0, None])
  # Vectorise along s parameter samples observations
  log_lik_vector_fun = jax.vmap(
      log_lik_vector_fun0, in_axes=[None, None, None, None, 0, 0])

  return log_lik_vector_fun(Z, Y, N, T, phi, theta)


# Joint distribution (data and params)
def log_prob_joint(
    batch: Batch,
    posterior_sample_dict: Dict[str, Any],
    smi_eta: Optional[SmiEta] = None,
) -> Array:
  """Compute the joint probability for the HPV model.

  The joint log probability of the model is given by

  .. math::

    log_prob(Z,Y,phi,theta) &= log_prob(Z \| \phi) \\
                            &+ log_prob(Y \| \phi, \theta)  \\
                            &+ log_prob(\phi)  \\
                            &+ log_prob(\theta).

  Optionally, if `smi_eta` is provided, the likelihoods are raised to the given
  power,

  .. math::

    log_prob(Z,Y,phi,theta) &= \eta_z * log_prob(Z \| \phi) \\
                            &+ \eta_y * log_prob(Y \| \phi, \theta) \\
                            &+ log_prob(\phi) \\
                            &+ log_prob(\theta).

  Args:
    batch: Dictionary with the data. Must contains 4 items 'Z','Y','T' and 'N',
      each one of shape (n,).
    posterior_sample_dict: Dictionary with values for the model parameters. Must
      contain 2 items: 'phi' and 'theta', arrays with shapes (s, n) and (s, 2),
      respectively.
    smi_eta: Optional dictionary with the power to be applied on each module of
      the likelihood. Must contain 'modules', an array of shape
      (1,2).

  Output:
    Array of shape (s,) with log joint probability evaluated on each value of
    the model parameters.
  """
  n_obs = batch['Z'].shape[0]

  num_samples, phi_dim = posterior_sample_dict['phi'].shape
  num_modules = 2

  # Batched log-likelihood
  loglik = log_lik_vectorised(batch['Z'], batch['Y'], batch['N'], batch['T'],
                              posterior_sample_dict['phi'],
                              posterior_sample_dict['theta'])
  assert loglik.shape == (num_samples, n_obs, num_modules)

  # If smi, multiply by eta
  if smi_eta is not None:
    loglik = loglik * jnp.broadcast_to(smi_eta['modules'], (1, 1, num_modules))

  # Add over observations and modules
  loglik = loglik.sum(axis=(1, 2))

  assert loglik.shape == (num_samples,)

  # Define priors
  log_prob_phi = distrax.Independent(
      tfd.Beta(
          concentration1=jnp.ones(phi_dim),
          concentration0=jnp.ones(phi_dim),
      ),
      reinterpreted_batch_ndims=1).log_prob
  log_prob_theta1 = distrax.Normal(loc=0., scale=100.).log_prob
  # log_prob_theta2 = tfd.TruncatedNormal(
  #     loc=0., scale=100., low=0.0, high=jnp.inf).log_prob
  log_prob_theta2 = tfd.Gamma(concentration=1, rate=0.1).log_prob

  # Everything together
  log_prob = (
      loglik + log_prob_phi(posterior_sample_dict['phi']) +
      log_prob_theta1(posterior_sample_dict['theta'][:, 0]) +
      log_prob_theta2(posterior_sample_dict['theta'][:, 1]))
  assert log_prob.shape == (num_samples,)

  return log_prob
