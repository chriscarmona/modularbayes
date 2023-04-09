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
    field_names=('phi', 'theta0', 'theta1'),
)
ModelParamsNoCut = namedtuple(
    'model_params_nocut',
    field_names=('phi',),
)
ModelParamsCut = namedtuple(
    "model_params_cut",
    field_names=('theta0', 'theta1'),
)
SmiEta = namedtuple(
    "smi_eta",
    field_names=('hpv', 'cancer'),
    defaults=(1., 1.),
)


# Joint distribution (data and params)
def logprob_joint(
    batch: Batch,
    model_params: ModelParams,
    prior_hparams: Optional[Dict[str, float]] = None,
    smi_eta: Optional[SmiEta] = None,
) -> float:
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
      each one of shape (n,), giving data for n populations.
    phi: Array with shape(n,), giving HPV prevalence for n populations.
    theta: Array with shape (2,), giving the intercept and slope of the
      log-linear relationship between HPV prevalence and cancer incidence.
    smi_eta: Optional named tuple, with the power to be applied on each module of
      the likelihood.

  Output:
    float giving the log joint probability evaluated on the model parameters.
  """

  # Set default arguments
  if prior_hparams is None:
    prior_hparams = {
        'phi_alpha': 1.,
        'phi_beta': 1.,
        'theta0_scale': 100.,
        'theta1_concentration': 1,
        'theta1_rate': 0.1,
    }
  if smi_eta is None:
    smi_eta = SmiEta()

  # Number of observations (one per population) in the data
  n_obs = batch['Z'].shape[0]

  phi = model_params.phi
  theta0 = model_params.theta0
  theta1 = model_params.theta1

  assert all(k in batch for k in ['Z', 'Y', 'T', 'N'])
  assert all(batch[k].shape == (n_obs,) for k in ['Z', 'Y', 'T', 'N'])
  assert phi.shape == (n_obs,)

  ### Define loglikelihood functions ###
  def log_prob_z_given_phi(phi: Array) -> float:
    """HPV prevalence model"""
    log_prob_ = distrax.Independent(
        tfd.Binomial(total_count=batch['N'], probs=phi),
        reinterpreted_batch_ndims=1).log_prob(batch['Z'])

    return log_prob_

  def log_prob_y_given_phi_theta(phi: Array, theta0: float,
                                 theta1: float) -> float:
    """Cancer incidence loglinear model"""
    log_incidence = theta0 + theta1 * phi
    mu = batch['T'] * (1. / 1000) * jnp.exp(log_incidence)
    log_prob_ = distrax.Independent(
        tfd.Poisson(rate=mu), reinterpreted_batch_ndims=1).log_prob(batch['Y'])
    return log_prob_

  ### Define priors functions ###
  log_prob_phi = distrax.Independent(
      distrax.Beta(
          alpha=prior_hparams['phi_alpha'] * jnp.ones(n_obs),
          beta=prior_hparams['phi_beta'] * jnp.ones(n_obs),
      ),
      reinterpreted_batch_ndims=1).log_prob
  log_prob_theta0 = distrax.Independent(
      distrax.Normal(loc=[0.], scale=[prior_hparams['theta0_scale']]),
      reinterpreted_batch_ndims=1).log_prob
  # log_prob_theta1 = tfd.TruncatedNormal(
  #     loc=0., scale=100., low=0.0, high=jnp.inf).log_prob
  log_prob_theta1 = distrax.Independent(
      distrax.Gamma(
          concentration=[prior_hparams['theta1_concentration']],
          rate=[prior_hparams['theta1_rate']]),
      reinterpreted_batch_ndims=1).log_prob

  log_prob = (
      log_prob_z_given_phi(phi) * smi_eta.hpv +
      log_prob_y_given_phi_theta(phi, theta0, theta1) * smi_eta.cancer +
      log_prob_phi(phi) + log_prob_theta0(theta0) + log_prob_theta1(theta1))

  return log_prob


def sample_eta_values(
    prng_key: PRNGKey,
    num_samples: int,
    eta_sampling_a: float,
    eta_sampling_b: float,
) -> SmiEta:
  """Generate a sample of the smi_eta values applicable to the model."""
  smi_etas = SmiEta(
      hpv=jnp.ones((num_samples,)),
      cancer=jax.random.beta(
          key=prng_key,
          a=eta_sampling_a,
          b=eta_sampling_b,
          shape=(num_samples,)),
  )
  return smi_etas