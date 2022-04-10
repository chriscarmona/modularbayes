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

  config.method = 'flow'

  # Defined in `flows`.
  config.flow_name = 'nsf'

  # kwargs to be passed to the flow
  config.flow_kwargs = ml_collections.ConfigDict()
  # Number of layers to use in the flow.
  config.flow_kwargs.num_layers = 8
  # Hidden sizes of the MLP conditioner.
  config.flow_kwargs.hidden_sizes = [30] * 3 + [10]
  # Number of bins to use in the rational-quadratic spline.
  config.flow_kwargs.num_bins = 10
  # the lower bound of the spline's range
  config.flow_kwargs.range_min = -10.
  # the upper bound of the spline's range
  config.flow_kwargs.range_max = 40.

  # SMI degree of influence
  config.smi_eta_groups = [1., 1.] + [1. for _ in range(config.num_groups - 2)]
  config.flow_kwargs.smi_eta = {'groups': config.smi_eta_groups}
  config.plot_suffix = 'full'

  # Number of samples used to estimate the ELBO.
  config.num_samples_elbo = 200

  # Number of training steps to run.
  config.training_steps = 30_000

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      'init_value': 0.,
      'peak_value': 2e-3,
      'warmup_steps': 2_000,
      'transition_steps': 10_000,
      'decay_rate': 0.5,
      'transition_begin': 0,
      'staircase': False,
      'end_value': None,
  }

  # How often to evaluate the model.
  config.eval_steps = int(config.training_steps / 10)

  # Initial seed for random numbers.
  config.seed = 0

  # How often to log images to monitor convergence.
  config.log_img_steps = int(config.training_steps / 10)

  # Number of posteriors samples used in the plots.
  config.num_samples_plot = 10_000

  config.num_samples_eval = 1_000

  # How often to save model checkpoints.
  config.checkpoint_steps = int(config.training_steps / 4)

  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  config.num_samples_log_prob_test = 10_000

  return config
