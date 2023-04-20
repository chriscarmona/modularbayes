"""Training a Variational Meta-Posterior, using the VMP map."""

import pathlib

from absl import logging

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax

from tensorflow_probability.substrates import jax as tfp

import flows
from flows import split_flow_nocut, split_flow_cut
import plot
from train_flow import load_data, sample_q_as_az, elpd_waic, elpd_truth_mc
from log_prob_fun import (ModelParams, ModelParamsCut, SmiEta, logprob_joint,
                          sample_eta_values)

import modularbayes
from modularbayes import (sample_q, sample_q_nocut, sample_q_cutgivennocut,
                          elbo_smi_vmpmap)
from modularbayes import (flatten_dict, initial_state_ckpt, update_states,
                          save_checkpoint, plot_to_image, normalize_images)
from modularbayes._src.utils.training import TrainState
from modularbayes._src.typing import (Any, Array, Callable, ConfigDict, Dict,
                                      List, Optional, PRNGKey, Tuple)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def make_optimizer(
    lr_schedule_name,
    lr_schedule_kwargs,
    grad_clip_value,
) -> optax.GradientTransformation:
  """Define optimizer to train the VHP map."""
  schedule = getattr(optax, lr_schedule_name)(**lr_schedule_kwargs)

  optimizer = optax.chain(*[
      optax.clip_by_global_norm(max_norm=grad_clip_value),
      optax.adabelief(learning_rate=schedule),
  ])
  return optimizer


def loss(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  # Compute ELBO.
  elbo_dict = elbo_smi_vmpmap(alpha_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2']))

  return loss_avg


def plot_elpd_surface(
    state_list: List[TrainState],
    dataset: Dict[str, Any],
    prng_key: PRNGKey,
    config: ConfigDict,
    vmpmap_fn: hk.Transformed,
    lambda_init_list: List[hk.Params],
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
  sample_q_jit = jax.jit(lambda x: sample_q(
      lambda_tuple=x,
      prng_key=sample_q_key_,  # same seed for all etas, see less variability
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      flow_kwargs=config.flow_kwargs,
      model_params_tupleclass=ModelParams,
      split_flow_fn_nocut=split_flow_nocut,
      split_flow_fn_cut=split_flow_cut,
      sample_shape=(config.num_samples_elpd,),
  )['model_params_sample'])

  # Produce flow parameters as a function of eta
  lambda_tuple = [
      vmpmap_fn.apply(
          state_i.params,
          eta_values=eta_grid.reshape(-1, num_groups),
          lambda_init=lambda_i,
      ) for state_i, lambda_i in zip(state_list, lambda_init_list)
  ]
  if true_params is not None:
    # Generate data from true model
    z = jax.random.normal(
        key=next(prng_seq), shape=(config.num_samples_elpd, num_groups))
    y_new = true_params.sigma * z + true_params.beta
  elpd_waic_grid = []
  if true_params is not None:
    elpd_grid = []
  for i, eta_i in enumerate(eta_grid.reshape(-1, num_groups)):
    # Sample from flow
    model_params_sample_i = sample_q_jit(
        jax.tree_map(lambda x: x[i], lambda_tuple))

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
    state_list: Tuple[TrainState],
    prng_key: PRNGKey,
    config: ConfigDict,
    dataset: Dict[str, Any],
    num_samples_plot: int,
    vmpmap_fn: hk.Transformed,
    lambda_init_list: List[hk.Params],
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

  # We can obtain the variational parameters for all eta values at once
  smi_etas = SmiEta(groups=jnp.array(config.smi_eta_plot.values()))
  eta_values = (
      smi_etas[0] if len(smi_etas) == 1 else jnp.stack(smi_etas, axis=-1))
  # Produce flow parameters as a function of eta
  lambda_tuple = [
      vmpmap_fn.apply(
          state_i.params,
          eta_values=eta_values,
          lambda_init=lambda_i,
      ) for state_i, lambda_i in zip(state_list, lambda_init_list)
  ]

  # Sample from flows and plot, one eta at a time
  for i, suffix in enumerate(config.smi_eta_plot.keys()):
    az_data = sample_q_as_az(
        lambda_tuple=jax.tree_map(lambda x: x[i], lambda_tuple),
        dataset=dataset,
        prng_key=next(prng_seq),
        flow_get_fn_nocut=flow_get_fn_nocut,
        flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
        flow_kwargs=config.flow_kwargs,
        sample_shape=(num_samples_plot,),
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
        np.array([0., 0.] + [1. for _ in range(config.num_groups)]),
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
        state_list=state_list,
        dataset=dataset,
        prng_key=next(prng_seq),
        config=config,
        vmpmap_fn=vmpmap_fn,
        lambda_init_list=lambda_init_list,
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
        state_list=state_list,
        dataset=dataset,
        prng_key=next(prng_seq),
        config=config,
        vmpmap_fn=vmpmap_fn,
        lambda_init_list=lambda_init_list,
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

  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')
  state_list = []
  state_name_list = []
  lambda_init_list = []

  # To initialize the VMP-map, we need one example of its output
  # The output of the VMP-map is lambda, the parameters of the variational posterior
  state_name_list.append('alpha_nocut')
  lambda_init_list.append(
      sample_q_nocut.init(
          next(prng_seq),
          flow_get_fn=flow_get_fn_nocut,
          flow_kwargs=config.flow_kwargs,
          split_flow_fn=split_flow_nocut,
          sample_shape=(config.num_samples_elbo,),
      ))
  nocut_base_sample_init = sample_q_nocut.apply(
      lambda_init_list[0],
      next(prng_seq),
      flow_get_fn=flow_get_fn_nocut,
      flow_kwargs=config.flow_kwargs,
      split_flow_fn=split_flow_nocut,
      sample_shape=(config.num_samples_elbo,),
  )['sample_base']
  state_name_list.append('alpha_cut')
  lambda_init_list.append(
      sample_q_cutgivennocut.init(
          next(prng_seq),
          flow_get_fn=flow_get_fn_cutgivennocut,
          flow_kwargs=config.flow_kwargs,
          split_flow_fn=split_flow_cut,
          nocut_base_sample=nocut_base_sample_init,
      ))
  state_name_list.append('alpha_cut_aux')
  lambda_init_list.append(
      sample_q_cutgivennocut.init(
          next(prng_seq),
          flow_get_fn=flow_get_fn_cutgivennocut,
          flow_kwargs=config.flow_kwargs,
          split_flow_fn=split_flow_cut,
          nocut_base_sample=nocut_base_sample_init,
      ))

  # Define function that produce a tuple of lambda (flow parameters)
  @hk.without_apply_rng
  @hk.transform
  def vmpmap_fn(eta_values, lambda_init):
    vmpmap = getattr(modularbayes, config.vmp_map_name)(
        **config.vmp_map_kwargs, params_flow_init=lambda_init)
    lambda_out = vmpmap(eta_values)
    return lambda_out

  ### Initialise Variational Meta-Posterior Map ###
  state_list = [
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
          forward_fn=vmpmap_fn,
          forward_fn_kwargs={
              'eta_values':
                  jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
              'lambda_init':
                  lambda_init_i
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      )
      for state_name_i, lambda_init_i in zip(state_name_list, lambda_init_list)
  ]
  # globals().update(forward_fn_kwargs)

  # Print a useful summary of the execution of the VHP-map architecture.
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda state_i, lambda_init_i: vmpmap_fn.apply(
          state_i.params,
          eta_values=jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
          lambda_init=lambda_init_i),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_list[0], lambda_init_list[0])
  logging.info('VMP-MAP no-cut parameters:')
  for line in summary.split("\n"):
    logging.info(line)

  summary = tabulate_fn_(state_list[1], lambda_init_list[1])
  logging.info('VMP-MAP cut parameters:')
  for line in summary.split("\n"):
    logging.info(line)

  ### Training VMP map ###
  @jax.jit
  def update_states_jit(state_list, batch, prng_key):
    return update_states(
        state_list=state_list,
        batch=batch,
        prng_key=prng_key,
        optimizer=make_optimizer(**config.optim_kwargs),
        loss_fn=loss,
        loss_fn_kwargs={
            'num_samples': config.num_samples_elbo,
            'vmpmap_fn': vmpmap_fn,
            'lambda_init_tuple': tuple(lambda_init_list),
            'sample_eta_fn': sample_eta_values,
            'sample_eta_kwargs': {
                'num_groups': config.num_groups,
                'eta_sampling_a': config.eta_sampling_a,
                'eta_sampling_b': config.eta_sampling_b
            },
            'elbo_smi_kwargs': {
                'logprob_joint_fn': logprob_joint,
                'flow_get_fn_nocut': flow_get_fn_nocut,
                'flow_get_fn_cutgivennocut': flow_get_fn_cutgivennocut,
                'flow_kwargs': config.flow_kwargs,
                'prior_hparams': config.prior_hparams,
                'model_params_tupleclass': ModelParams,
                'model_params_cut_tupleclass': ModelParamsCut,
                'split_flow_fn_nocut': split_flow_nocut,
                'split_flow_fn_cut': split_flow_cut,
            }
        },
    )

  if state_list[0].step < config.training_steps:
    logging.info('Training Variational Meta-Posterior (VMP-map)...')

  # Reset random key sequence
  prng_seq = hk.PRNGSequence(config.seed)

  while state_list[0].step < config.training_steps:

    # Plots to monitor training
    if (config.log_img_steps
        is not None) and (state_list[0].step % config.log_img_steps == 0):
      logging.info("\t\t Logging plots...")
      log_images(
          state_list=state_list,
          prng_key=next(prng_seq),
          config=config,
          dataset=train_ds,
          num_samples_plot=config.num_samples_plot,
          vmpmap_fn=vmpmap_fn,
          lambda_init_list=lambda_init_list,
          flow_get_fn_nocut=flow_get_fn_nocut,
          flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
          show_elpd_surface=True,
          true_params=true_params,
          summary_writer=summary_writer,
          workdir_png=workdir,
      )
      plt.close()
      logging.info("\t\t...done logging plots.")

    # Log learning rate
    summary_writer.scalar(
        tag='learning_rate',
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(state_list[0].step),
        step=state_list[0].step,
    )

    state_list, metrics = update_states_jit(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
    )

    # The computed training loss would correspond to the model before update
    summary_writer.scalar(
        tag='train_loss',
        value=metrics['train_loss'],
        step=state_list[0].step - 1,
    )

    if state_list[0].step == 2:
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

    if state_list[0].step % config.eval_steps == 0:
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

    # Save checkpoints
    if (state_list[0].step) % config.checkpoint_steps == 0:
      for state_i, state_name_i in zip(state_list, state_name_list):
        save_checkpoint(
            state=state_i,
            checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
            keep=config.checkpoints_keep,
        )

    # Wait until computations are done before the next step
    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  logging.info('Final training step: %i', state_list[0].step)

  # Save checkpoints at the end of the training process
  # (in case training_steps is not multiple of checkpoint_steps)
  for state_i, state_name_i in zip(state_list, state_name_list):
    save_checkpoint(
        state=state_i,
        checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
        keep=config.checkpoints_keep,
    )

  # Last plot of posteriors
  log_images(
      state_list=state_list,
      prng_key=next(prng_seq),
      config=config,
      dataset=train_ds,
      num_samples_plot=config.num_samples_plot,
      vmpmap_fn=vmpmap_fn,
      lambda_init_list=lambda_init_list,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      show_elpd_surface=True,
      true_params=true_params,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )
  plt.close()

  return state_list


# # For debugging
# config = get_config()
# import pathlib
# workdir = str(pathlib.Path.home() / f'modularbayes-output-exp/random_effects/nsf/vmp_map')
# # train_and_evaluate(config, workdir)
