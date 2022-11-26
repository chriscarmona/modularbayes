"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'flow'

  # SMI degree of influence of the poisson module
  config.smi_eta = 1.0

  # Defined in `epidemiology.models.flows`.
  config.flow_name = 'mean_field'

  # kwargs to be passed to the flow
  config.flow_kwargs = ml_collections.ConfigDict()

  # Number of posteriors samples to approximate the variational loss (ELBO).
  config.num_samples_elbo = 100

  # Number of training steps to run.
  config.training_steps = 2_000

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      'init_value': 0.,
      'peak_value': 1e-2,
      'warmup_steps': 1_000,
      'transition_steps': config.training_steps / 2,
      'decay_rate': 0.25,
      'transition_begin': 0,
      'staircase': False,
      'end_value': None,
  }

  # How often to evaluate the model.
  config.eval_steps = config.training_steps / 10
  config.num_samples_eval = 5_000

  # Initial seed for random numbers.
  config.seed = 0

  # How often to log images to monitor convergence.
  config.log_img_steps = config.training_steps / 10

  # Number of posteriors samples used in the plots.
  config.num_samples_plot = 10_000

  # How often to save model checkpoints.
  config.checkpoint_steps = config.training_steps / 4

  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  config.num_samples_log_prob_test = 10_000

  return config
