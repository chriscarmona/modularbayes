"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'vmp_map'

  # Defined in `epidemiology.models.flows`.
  config.flow_name = 'mean_field'

  # kwargs to be passed to the flow
  config.flow_kwargs = ml_collections.ConfigDict()

  # Number of posteriors samples to approximate the variational loss (ELBO).
  config.num_samples_elbo = 50

  # Number of training steps to run.
  config.training_steps = 20000

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      'init_value': 0.,
      'peak_value': 1e-2,
      'warmup_steps': 1_000,
      'transition_steps': config.training_steps,
      'decay_rate': 0.5,
      'transition_begin': 0,
      'staircase': False,
      'end_value': None,
  }

  # How often to evaluate the model.
  config.eval_steps = config.training_steps / 20
  config.num_samples_eval = 5_000

  # Initial seed for random numbers.
  config.seed = 0

  # How often to log images to monitor convergence.
  config.log_img_steps = config.training_steps / 10

  # Number of posteriors samples used in the plots.
  config.num_samples_plot = 1000

  config.eta_plot = (0.001, 0.1, 1.0)

  # How often to save model checkpoints.
  config.checkpoint_steps = config.training_steps / 4

  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  # Arguments for the Variational Meta-Posterior map
  config.vmp_map_name = 'VmpCubicSpline'
  config.vmp_map_kwargs = ml_collections.ConfigDict()
  config.vmp_map_kwargs.num_knots = 5

  # Number of samples of eta for Meta-Posterior training
  config.num_samples_eta = 25
  config.eta_sampling_a = 0.2
  config.eta_sampling_b = 0.2

  config.lambda_idx_plot = [5 * i for i in range(5)]
  config.constant_lambda_ignore_plot = False

  config.state_flow_init_path = ''
  config.eps_noise_pretrain = 1e-2
  config.pretrain_error = 1.

  return config
