"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.method = 'vmp_map'

  # Defined in `epidemiology.models.flows`.
  config.flow_name = 'nsf'

  # kwargs to be passed to the flow
  config.flow_kwargs = ml_collections.ConfigDict()
  # Number of layers to use in the flow.
  config.flow_kwargs.num_layers = 4
  # Hidden sizes of the MLP conditioner.
  config.flow_kwargs.hidden_sizes = [5] * 3
  # Number of bins to use in the rational-quadratic spline.
  config.flow_kwargs.num_bins = 10
  # the lower bound of the spline's range
  config.flow_kwargs.range_min = -10.
  # the upper bound of the spline's range
  config.flow_kwargs.range_max = 40.

  # Number of samples to approximate ELBO's gradient
  config.num_samples_elbo = 100

  # Number of training steps to run.
  config.training_steps = 30_000

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      'init_value': 0.,
      'peak_value': 1e-3,
      'warmup_steps': 3_000,
      'transition_steps': config.training_steps / 4,
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
  config.num_samples_plot = 10_000

  config.eta_plot = [0.001, 0.1, 1.0]

  # How often to save model checkpoints.
  config.checkpoint_steps = config.training_steps / 4

  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  # Arguments for the Variational Meta-Posterior map
  config.vmp_map_name = 'VmpMLP'
  config.vmp_map_kwargs = ml_collections.ConfigDict()
  eta_dim = 1
  config.vmp_map_kwargs.hidden_sizes = [
      eta_dim * 2**i for i in sorted(list(range(1, 7)) * 3)
  ] + [5]

  # Number of samples of eta for Meta-Posterior training
  config.num_samples_eta = 25
  config.eta_sampling_a = 0.2
  config.eta_sampling_b = 0.2

  config.lambda_idx_plot = [50 * i for i in range(5)]
  config.constant_lambda_ignore_plot = True

  config.state_flow_init_path = ''
  config.eps_noise_pretrain = 0.
  config.pretrain_error = 1.

  return config
