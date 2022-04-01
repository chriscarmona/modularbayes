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

  config.method = 'vmp_map'

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

  config.num_samples_elbo = 100
  config.num_samples_eval = 5_000

  # Number of training steps to run.
  config.training_steps = 50_000

  # Optimizer for pretrain
  config.optim_pretrain_kwargs = ml_collections.ConfigDict()
  config.optim_pretrain_kwargs.grad_clip_value = 1.0
  config.optim_pretrain_kwargs.learning_rate = 3e-4

  # Optimizer
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  # Using SGD with warm restarts, from Loschilov & Hutter (arXiv:1608.03983).
  config.optim_kwargs.lr_schedule_name = 'warmup_exponential_decay_schedule'
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      'init_value': 0.,
      'peak_value': 3e-4,
      'warmup_steps': 3_000,
      'transition_steps': 10_000,
      'decay_rate': 0.5,
      'transition_begin': 0,
      'staircase': False,
      'end_value': None,
  }

  # How often to evaluate the model.
  config.eval_steps = int(config.training_steps / 20)

  # Initial seed for random numbers.
  config.seed = 123

  # How often to log images to monitor convergence.
  config.log_img_steps = int(config.training_steps / 10)

  # Number of posteriors samples used in the plots.
  config.num_samples_plot = 10_000

  config.eta_plot = [
      [1. for _ in range(config.num_groups)],
      [0.0001, 1.] + [1. for _ in range(config.num_groups - 2)],
      [0.0001, 0.0001] + [1. for _ in range(config.num_groups - 2)],
  ]
  config.suffix_eta_plot = ['full', 'cut1', 'cut2']

  # How often to save model checkpoints.
  config.checkpoint_steps = int(config.training_steps / 4)

  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  # Arguments for the Variational Meta-Posterior map
  config.vmp_map_name = 'VmpMLP'
  config.vmp_map_kwargs = ml_collections.ConfigDict()
  eta_dim = config.num_groups
  config.vmp_map_kwargs.hidden_sizes = [eta_dim * 10] * 5 + [20]
  # config.vmp_map_kwargs.hidden_sizes = [
  #     eta_dim * 2**i for i in sorted(list(range(1, 5)) * 2)
  # ] + [10]

  # Number of samples of eta for Meta-Posterior training
  config.num_samples_eta = 25
  config.eta_sampling_a = 0.2
  config.eta_sampling_b = 1.0

  config.lambda_idx_plot = [50 * i for i in range(5)]
  config.constant_lambda_ignore_plot = True

  config.pretrain_vmp_map = False
  config.state_sigma_init = ''
  config.state_beta_tau_init = ''
  config.state_beta_tau_aux_init = ''
  config.eps_noise_pretrain = 1e-5
  config.pretrain_error = 100.

  return config
