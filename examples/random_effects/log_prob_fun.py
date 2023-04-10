"""Probability functions for the Epidemiology model."""

from typing import Dict, Optional
from collections import namedtuple

import jax
import jax.numpy as jnp
import distrax
from tensorflow_probability.substrates import jax as tfp

from modularbayes._src.typing import Array, Batch, PRNGKey

tfd = tfp.distributions

ModelParams = namedtuple(
    "model_params",
    field_names=('sigma', 'beta', 'tau'),
)
ModelParamsNoCut = namedtuple(
    'model_params_nocut',
    field_names=('sigma',),
)
ModelParamsCut = namedtuple(
    "model_params_cut",
    field_names=('beta', 'tau'),
)
SmiEta = namedtuple(
    "smi_eta",
    field_names=('groups'),
    defaults=(1.0,),
)


# Joint distribution (data and params)
def logprob_joint(
    batch: Batch,
    model_params: ModelParams,
    prior_hparams=None,
    smi_eta: Optional[SmiEta] = None,
) -> float:
  """Compute the joint probability for the Random effects model.

  The joint log probability of the model is given by

  .. math::

    log_prob(Y,tau,beta,sigma) &= \prod_{i=1}^{N} log_prob(Y_i \| \beta, \sigma) \\
                            &+ log_prob(beta_i \| \tau)  \\
                            &+ log_prob(\tau \| \sigma)  \\
                            &+ log_prob(\sigma).
  """
  if smi_eta is None:
    smi_eta = SmiEta()

  ### Define loglikelihood function ###
  def log_prob_y_given_betasigma(beta: Array, sigma: Array) -> float:
    return distrax.Independent(
        distrax.Normal(loc=beta[batch['group']], scale=sigma[batch['group']]),
        reinterpreted_batch_ndims=1).log_prob(batch['Y'])

  # Define priors
  def log_prob_beta_given_tau(beta: Array, tau: Array,
                              eta_groups: Array) -> Array:
    return distrax.Independent(
        distrax.Normal(loc=0, scale=tau / eta_groups),
        reinterpreted_batch_ndims=1).log_prob(beta)

  # If smi, multiply by eta

  def log_prob_tau(tau: Array, sigma: Array, num_obs_groups: Array) -> float:
    log_prob_ = -jnp.log(tau**2 + (sigma**2 / num_obs_groups).mean())
    assert log_prob_.shape == (1,)
    return log_prob_[0]

  def log_prob_sigma(sigma: Array):
    return -2 * jnp.log(sigma).sum()

  # Everything together
  log_prob = (
      log_prob_y_given_betasigma(
          beta=model_params.beta, sigma=model_params.sigma) +
      log_prob_beta_given_tau(
          beta=model_params.beta,
          tau=model_params.tau,
          eta_groups=smi_eta.groups) + log_prob_tau(
              tau=model_params.tau,
              sigma=model_params.sigma,
              num_obs_groups=batch['num_obs_groups']) +
      log_prob_sigma(sigma=model_params.sigma))

  return log_prob


def sample_eta_values(
    prng_key: PRNGKey,
    num_samples: int,
    num_groups: int,
    eta_sampling_a: float,
    eta_sampling_b: float,
) -> SmiEta:
  """Generate a sample of the smi_eta values applicable to the model."""
  smi_etas = SmiEta(
      groups=jax.random.beta(
          key=prng_key,
          a=eta_sampling_a,
          b=eta_sampling_b,
          shape=(num_samples, num_groups)),)
  return smi_etas