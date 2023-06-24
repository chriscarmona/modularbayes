"""Training a Variational Meta-Posterior, using the VMP flow."""

import pathlib

from absl import logging

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import jax
from jax import numpy as jnp
from flax.metrics import tensorboard
from flax.training.train_state import TrainState
import haiku as hk
import optax
import orbax.checkpoint

import flows
from flows import split_flow_nocut, split_flow_cut
from log_prob_fun import (ModelParams, ModelParamsCut, SmiEta, logprob_joint,
                          sample_eta_values)
import plot
from train_flow import (elpd_truth_mc, elpd_waic, init_state_tuple, load_data,
                        print_trainable_params, sample_q_as_az)
from modularbayes import elbo_smi_vmpflow, sample_q, train_step
from modularbayes import (flatten_dict, normalize_images, plot_to_image)
from modularbayes._src.typing import (Any, Array, Callable, ConfigDict, Dict,
                                      List, Optional, PRNGKey, Tuple)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def loss_fn(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  ### Compute ELBO ###
  elbo_dict = elbo_smi_vmpflow(lambda_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2']))

  return loss_avg


def plot_elpd_surface(
    lambda_tuple: Tuple[hk.Params],
    dataset: Dict[str, Any],
    prng_key: PRNGKey,
    config: ConfigDict,
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    eta_grid: Array,
    eta_grid_x_y_idx: Tuple[int, int],
    true_params: Optional[ModelParams] = None,
):
  """Visualize ELPD surface as function of eta."""

  assert eta_grid.ndim == 3
  prng_seq = hk.PRNGSequence(prng_key)

  num_groups = eta_grid.shape[-1]

  # Jit functions for speed
  elpd_waic_jit = jax.jit(elpd_waic)
  elpd_truth_mc_jit = jax.jit(elpd_truth_mc)
  sample_q_key_ = next(prng_seq)
  sample_q_jit = jax.jit(lambda x, y: sample_q(
      lambda_tuple=x,
      prng_key=sample_q_key_,  # same seed for all etas, see less variability
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      flow_kwargs=config.flow_kwargs,
      model_params_tupleclass=ModelParams,
      split_flow_fn_nocut=split_flow_nocut,
      split_flow_fn_cut=split_flow_cut,
      sample_shape=(y.shape[0],),
      eta_values=y,
  )['model_params_sample'])

  # Produce flow parameters as a function of eta
  if true_params is not None:
    # Generate data from true model
    z = jax.random.normal(
        key=next(prng_seq), shape=(config.num_samples_elpd, num_groups))
    y_new = true_params.sigma * z + true_params.beta
  elpd_waic_grid = []
  if true_params is not None:
    elpd_grid = []
  for eta_i in eta_grid.reshape(-1, num_groups):
    # Sample from flow
    model_params_sample_i = sample_q_jit(
        lambda_tuple, jnp.tile(eta_i[None, ...], (config.num_samples_elpd, 1)))

    # Compute ELPD using WAIC
    elpd_waic_grid.append(
        elpd_waic_jit(
            model_params_sample=model_params_sample_i, dataset=dataset))
    if true_params is not None:
      # Compute ELPD
      elpd_grid.append(
          elpd_truth_mc_jit(
              model_params_sample=model_params_sample_i, y_new=y_new))

  # Plot the ELPD surface.
  fig, axs = plt.subplots(
      nrows=1,
      ncols=1 if true_params is None else 2,
      figsize=(4 if true_params is None else 8, 3),
      subplot_kw={"projection": "3d"},
      squeeze=False)
  elpd_waic_grid = jnp.array(elpd_waic_grid).reshape(eta_grid.shape[:-1])
  axs[0, 0].plot_surface(
      eta_grid[..., eta_grid_x_y_idx[0]],
      eta_grid[..., eta_grid_x_y_idx[1]],
      -elpd_waic_grid,
      cmap=matplotlib.cm.inferno,
      # linewidth=0,
      # antialiased=False,
  )
  axs[0, 0].view_init(30, 225)
  axs[0, 0].set_xlabel(f'eta_{eta_grid_x_y_idx[0]}')
  axs[0, 0].set_ylabel(f'eta_{eta_grid_x_y_idx[1]}')
  axs[0, 0].set_title('- ELPD WAIC')
  if true_params is not None:
    elpd_grid = jnp.array(elpd_grid).reshape(eta_grid.shape[:-1])
    axs[0, 1].plot_surface(
        eta_grid[..., eta_grid_x_y_idx[0]],
        eta_grid[..., eta_grid_x_y_idx[1]],
        -elpd_grid,
        cmap=matplotlib.cm.inferno,
        # linewidth=0,
        # antialiased=False,
    )
    axs[0, 1].view_init(30, 225)
    axs[0, 1].set_xlabel(f'eta_{eta_grid_x_y_idx[0]}')
    axs[0, 1].set_ylabel(f'eta_{eta_grid_x_y_idx[1]}')
    axs[0, 1].set_title('- ELPD')
  fig.tight_layout()

  return fig, axs


def log_images(
    state_list: List[TrainState],
    prng_key: PRNGKey,
    config: ConfigDict,
    dataset: Dict[str, Any],
    num_samples_plot: int,
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    show_elpd_surface: bool,
    true_params: ModelParams,
    summary_writer: Optional[tensorboard.SummaryWriter] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  model_name = 'random_effects'

  prng_seq = hk.PRNGSequence(prng_key)

  assert len(config.smi_eta_plot.keys()) > 0, 'No eta values to plot'

  # Plot posterior samples
  for suffix, eta_groups_i in config.smi_eta_plot.items():
    # Define eta
    eta_groups_i = jnp.array(eta_groups_i)
    assert all(eta_groups_i >= 0.0) and all(eta_groups_i <= 1.0), (
        'eta must be in [0, 1]')
    smi_etas = SmiEta(
        groups=jnp.broadcast_to(eta_groups_i,
                                (num_samples_plot,) + eta_groups_i.shape),)
    # Sample from flow
    az_data = sample_q_as_az(
        lambda_tuple=tuple(x.params for x in state_list),
        dataset=dataset,
        prng_key=next(prng_seq),
        flow_get_fn_nocut=flow_get_fn_nocut,
        flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
        flow_kwargs=config.flow_kwargs,
        sample_shape=(num_samples_plot,),
        eta_values=(smi_etas[0] if len(smi_etas) == 1 else jnp.stack(
            smi_etas, axis=-1)),
    )
    plot.posterior_plots(
        az_data=az_data,
        show_sigma_trace=False,
        show_beta_trace=False,
        show_tau_trace=False,
        betasigma_pairplot_groups=(0, 1, 2),
        tausigma_pairplot_groups=(0, 1, 2),
        suffix=f'_{suffix}',
        step=state_list[0].step,
        workdir_png=workdir_png,
        summary_writer=summary_writer,
    )

  if show_elpd_surface:
    eta_grid_len = 10
    images = []

    # Define elements to grate grid of eta values
    eta_grid_base = np.tile(
        np.array([0., 0.] + [1. for _ in range(config.num_groups - 2)]),
        [eta_grid_len + 1, eta_grid_len + 1, 1])
    eta_grid_mini = np.stack(
        np.meshgrid(
            np.linspace(0, 1, eta_grid_len + 1).round(5),
            np.linspace(0, 1, eta_grid_len + 1).round(5)),
        axis=0).T

    # Vary eta_0 and eta_1
    plot_name = f'{model_name}_elpd_surface_eta_0_1'
    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [0, 1]
    eta_grid[:, :, eta_grid_x_y_idx] = eta_grid_mini
    fig, _ = plot_elpd_surface(
        lambda_tuple=tuple(x.params for x in state_list),
        dataset=dataset,
        prng_key=next(prng_seq),
        config=config,
        flow_get_fn_nocut=flow_get_fn_nocut,
        flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
        eta_grid=eta_grid,
        eta_grid_x_y_idx=eta_grid_x_y_idx,
        true_params=true_params,
    )
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    # Vary eta_0 and eta_2
    plot_name = f'{model_name}_elpd_surface_eta_0_2'
    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [0, 2]
    eta_grid[:, :, eta_grid_x_y_idx] = eta_grid_mini
    fig, _ = plot_elpd_surface(
        lambda_tuple=tuple(x.params for x in state_list),
        dataset=dataset,
        prng_key=next(prng_seq),
        config=config,
        flow_get_fn_nocut=flow_get_fn_nocut,
        flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
        eta_grid=eta_grid,
        eta_grid_x_y_idx=eta_grid_x_y_idx,
        true_params=true_params,
    )
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    if summary_writer:
      plot_name = f"{model_name}_elpd_surface"
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=state_list[0].step,
          max_outputs=len(images),
      )


def train_and_evaluate(config: ConfigDict, workdir: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  # Add trailing slash
  workdir = workdir.rstrip("/") + '/'

  # Initialize random keys
  prng_seq = hk.PRNGSequence(config.seed)

  # True generative parameters
  true_params = ModelParams(
      beta=jnp.array(config.loc_groups),
      sigma=jnp.array(config.scale_groups),
      tau=None,
  )

  # Generate dataset
  train_ds = load_data(
      prng_key=next(prng_seq),
      num_obs_groups=config.num_obs_groups,
      loc_groups=true_params.beta,
      scale_groups=true_params.sigma,
  )

  # num_groups is also a parameter of the flow,
  # as it define its dimension
  config.flow_kwargs.num_groups = config.num_groups
  # Also is_smi modifies the dimension of the flow, due to the duplicated params
  config.flow_kwargs.is_smi = True

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)
  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))
  else:
    summary_writer = None

  # Functions that generate the NF for no-cut params
  flow_get_fn_nocut = getattr(flows, 'get_q_nocut_' + config.flow_name)
  # Functions that generate the NF for cut params (conditional on no-cut params)
  flow_get_fn_cutgivennocut = getattr(flows,
                                      'get_q_cutgivennocut_' + config.flow_name)

  # Initialize States of the three flows
  state_tuple = init_state_tuple(
      prng_key=next(prng_seq),
      config=config,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      sample_shape=(config.num_samples_elbo,),
      eta_values=jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
  )

  # Print a summary of the networks architecture
  print_trainable_params(
      state_tuple=state_tuple,
      prng_key=next(prng_seq),
      config=config,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      sample_shape=(config.num_samples_elbo,),
      eta_values=jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
  )

  # Create checkpoint managers for the three states
  orbax_ckpt_mngrs = [
      orbax.checkpoint.CheckpointManager(
          directory=str(pathlib.Path(workdir) / 'checkpoints' / state_name),
          checkpointers=orbax.checkpoint.PyTreeCheckpointer(),
          options=orbax.checkpoint.CheckpointManagerOptions(
              max_to_keep=1,
              save_interval_steps=config.checkpoint_steps,
          ),
      ) for state_name in state_tuple._asdict()
  ]

  # Restore existing checkpoint if present
  if orbax_ckpt_mngrs[0].latest_step() is not None:
    state_tuple = [
        mngr.restore(mngr.latest_step(), items=state)
        for state, mngr in zip(state_tuple, orbax_ckpt_mngrs)
    ]

  # Jit function to update training states
  @jax.jit
  def train_step_jit(state_tuple, batch, prng_key):
    return train_step(
        state_tuple=state_tuple,
        batch=batch,
        prng_key=prng_key,
        loss=loss_fn,
        loss_kwargs={
            'num_samples': config.num_samples_elbo,
            'logprob_joint_fn': logprob_joint,
            'flow_get_fn_nocut': flow_get_fn_nocut,
            'flow_get_fn_cutgivennocut': flow_get_fn_cutgivennocut,
            'flow_kwargs': config.flow_kwargs,
            'prior_hparams': config.prior_hparams,
            'model_params_tupleclass': ModelParams,
            'model_params_cut_tupleclass': ModelParamsCut,
            'split_flow_fn_nocut': split_flow_nocut,
            'split_flow_fn_cut': split_flow_cut,
            'sample_eta_fn': sample_eta_values,
            'sample_eta_kwargs': {
                'num_groups': config.num_groups,
                'eta_sampling_a': 1.,
                'eta_sampling_b': 1.
            },
        },
    )

  @jax.jit
  def elbo_validation_jit(state_list, batch, prng_key):
    return elbo_smi_vmpflow(
        lambda_tuple=tuple(state.params for state in state_list),
        batch=batch,
        prng_key=prng_key,
        num_samples=config.num_samples_eval,
        logprob_joint_fn=logprob_joint,
        flow_get_fn_nocut=flow_get_fn_nocut,
        flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
        flow_kwargs=config.flow_kwargs,
        prior_hparams=config.prior_hparams,
        model_params_tupleclass=ModelParams,
        model_params_cut_tupleclass=ModelParamsCut,
        split_flow_fn_nocut=split_flow_nocut,
        split_flow_fn_cut=split_flow_cut,
        sample_eta_fn=sample_eta_values,
        sample_eta_kwargs={
            'num_groups': config.num_groups,
            'eta_sampling_a': 1.,
            'eta_sampling_b': 1.
        },
    )

  if state_tuple[0].step < config.training_steps:
    logging.info('Training Variational Meta-Posterior (VMP-flow)...')

  # Reset random key sequence
  prng_seq = hk.PRNGSequence(config.seed)

  while state_tuple[0].step < config.training_steps:

    # Plots to monitor training
    if ((state_tuple[0].step == 0) or
        (state_tuple[0].step % config.log_img_steps == 0)):
      logging.info("\t\t Logging plots...")
      log_images(
          state_list=state_tuple,
          prng_key=next(prng_seq),
          config=config,
          dataset=train_ds,
          num_samples_plot=config.num_samples_plot,
          flow_get_fn_nocut=flow_get_fn_nocut,
          flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
          show_elpd_surface=True,
          true_params=true_params,
          summary_writer=summary_writer,
          workdir_png=workdir,
      )
      logging.info("\t\t...done logging plots.")

    # Log learning rate
    summary_writer.scalar(
        tag='learning_rate',
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(state_tuple[0].step),
        step=state_tuple[0].step,
    )

    # SGD step
    state_tuple_, metrics = train_step_jit(
        state_tuple=state_tuple,
        batch=train_ds,
        prng_key=next(prng_seq),
    )
    if jax.lax.is_finite(metrics['train_loss']):
      state_tuple = state_tuple_

    summary_writer.scalar(
        tag='train_loss',
        value=metrics['train_loss'],
        step=state_tuple[0].step,
    )

    if state_tuple[0].step == 1:
      logging.info("STEP: %5d; training loss: %.3f", state_tuple[0].step,
                   metrics["train_loss"])

    # Metrics for evaluation
    if state_tuple[0].step % config.eval_steps == 0:
      logging.info("STEP: %5d; training loss: %.3f", state_tuple[0].step,
                   metrics["train_loss"])

      elbo_dict = elbo_validation_jit(
          state_list=state_tuple,
          batch=train_ds,
          prng_key=next(prng_seq),
      )
      for k, v in elbo_dict.items():
        summary_writer.scalar(
            tag=f'elbo_{k}',
            value=v.mean(),
            step=state_tuple[0].step,
        )

    for state, mngr in zip(state_tuple, orbax_ckpt_mngrs):
      mngr.save(step=int(state.step), items=state)

  logging.info('Final training step: %i', state_tuple[0].step)

  # Last plot of posteriors
  log_images(
      state_list=state_tuple,
      prng_key=next(prng_seq),
      config=config,
      dataset=train_ds,
      num_samples_plot=config.num_samples_plot,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      show_elpd_surface=True,
      true_params=true_params,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )

  return state_tuple


# # For debugging
# config = get_config()
# import pathlib
# workdir = str(pathlib.Path.home() / f'modularbayes-output-exp/random_effects/nsf/vmp_flow')
# # train_and_evaluate(config, workdir)
