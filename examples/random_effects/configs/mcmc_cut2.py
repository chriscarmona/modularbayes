"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Data
  config.num_groups = 30
  config.num_obs_groups = [5, 5] + [5 for _ in range(config.num_groups - 2)]
  config.loc_groups = [10., 5.] + [0. for _ in range(config.num_groups - 2)]
  config.scale_groups = [1. for _ in range(config.num_groups)]

  # Model hyper-parameters, defining the prior.
  config.prior_hparams = None

  config.method = 'mcmc'

  config.num_chains = 4
  config.num_samples = 10_000
  config.num_samples_subchain_stg2 = 50
  config.num_samples_perchunk_stg2 = 1_000
  config.num_steps_call_warmup = 100

  config.smi_eta_groups = [0.001, 0.001
                          ] + [1. for _ in range(config.num_groups - 2)]
  config.plot_suffix = '_cut2'

  config.seed = 0

  return config
