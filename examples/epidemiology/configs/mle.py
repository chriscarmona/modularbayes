"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'mle'

  config.learning_rate = 1e-1
  config.training_steps = 1000
  config.eval_steps = int(config.training_steps / 20)

  config.seed = 0

  return config
