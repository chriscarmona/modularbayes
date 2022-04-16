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

  config.method = 'mcmc'

  # MCMC
  config.num_samples = 10000
  config.num_samples_subchain = 100
  config.num_burnin_steps = 2000
  config.mcmc_step_size = 0.01

  config.smi_eta = {
      'groups': [0.001, 0.001] + [1. for _ in range(config.num_groups - 2)],
  }
  config.plot_suffix = 'cut2'

  config.seed = 0

  return config
