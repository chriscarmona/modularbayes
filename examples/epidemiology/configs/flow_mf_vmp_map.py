"""Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Model hyper-parameters, defining the prior.
  config.prior_hparams = ml_collections.ConfigDict()
  config.prior_hparams.phi_alpha = 1.0
  config.prior_hparams.phi_beta = 1.0
  config.prior_hparams.theta0_scale = 100.0
  config.prior_hparams.theta1_concentration = 1
  config.prior_hparams.theta1_rate = 0.1

  config.method = "vmp_map"

  # Defined in `epidemiology.models.flows`.
  config.flow_name = "mf"

  # kwargs to be passed to the flow
  config.flow_kwargs = ml_collections.ConfigDict()

  # Number of posteriors samples to approximate the variational loss (ELBO).
  config.num_samples_elbo = 10

  # Number of training steps to run.
  config.training_steps = 10_000

  # Optimizer.
  config.optim_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.grad_clip_value = 1.0
  config.optim_kwargs.lr_schedule_name = "warmup_exponential_decay_schedule"
  config.optim_kwargs.lr_schedule_kwargs = ml_collections.ConfigDict()
  config.optim_kwargs.lr_schedule_kwargs = {
      "init_value": 0.0,
      "peak_value": 3e-3,
      "warmup_steps": 3_000,
      "transition_steps": 10_000,
      "decay_rate": 0.5,
      "transition_begin": 0,
      "staircase": False,
      "end_value": None,
  }

  # How often to evaluate the model.
  config.eval_steps = config.training_steps / 10
  config.num_samples_eval = 5_000

  # Initial seed for random numbers.
  config.seed = 0

  # How often to log images to monitor convergence.
  config.log_img_steps = config.training_steps / 10

  # Number of posteriors samples used in the plots.
  config.num_samples_plot = 40_000

  config.smi_eta_dim = 2
  config.smi_eta_cancer_plot = (0.001, 0.1, 0.2, 0.5, 1.0)

  # How often to save model checkpoints.
  config.checkpoint_steps = config.training_steps / 4

  # How many checkpoints to keep.
  config.checkpoints_keep = 1

  # Arguments for the Variational Meta-Posterior map
  config.vmp_map_name = "MLPVmpMap"
  config.vmp_map_kwargs = ml_collections.ConfigDict()
  config.vmp_map_kwargs.hidden_sizes = [10] * 5

  # Number of samples of eta for Meta-Posterior training
  config.eta_sampling_a = 0.2
  config.eta_sampling_b = 1.0

  config.vmpmap_curves_idx = [5 * i for i in range(5)]

  return config
