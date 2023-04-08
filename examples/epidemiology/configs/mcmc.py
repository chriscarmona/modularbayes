"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Model hyper-parameters, defining the prior.
  config.prior_hparams = ml_collections.ConfigDict()
  config.prior_hparams.phi_alpha = 1.
  config.prior_hparams.phi_beta = 1.
  config.prior_hparams.theta0_scale = 100.
  config.prior_hparams.theta1_concentration = 1
  config.prior_hparams.theta1_rate = 0.1

  config.method = 'mcmc'

  config.num_chains = 4
  config.num_samples = 10_000

  config.num_samples_subchain_stg2 = 50
  config.num_samples_perchunk_stg2 = 1_000
  config.num_steps_call_warmup = 100

  config.smi_eta_cancer = 1.0

  config.seed = 0

  return config
