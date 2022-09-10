"""A simple example of variational SMI on the Random Effects model."""

import math
import pathlib

from absl import logging

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import distrax
import optax

from train_flow import (get_dataset, sample_all_flows, q_distr_sigma,
                        q_distr_beta_tau, elbo_estimate, compute_elpd)
import plot

import modularbayes
from modularbayes import (flatten_dict, plot_to_image, normalize_images,
                          initial_state_ckpt, update_states, save_checkpoint)
from modularbayes._src.utils.training import TrainState
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict, List,
                                      Mapping, Optional, PRNGKey, SummaryWriter,
                                      Tuple, Union)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)

compute_elpd_jit = jax.jit(compute_elpd)


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


# Define Meta-Posterior map
# Produce flow parameters as a function of eta
@hk.without_apply_rng
@hk.transform
def vmp_map(eta, vmp_map_name, vmp_map_kwargs, params_flow_init):
  return getattr(modularbayes, vmp_map_name)(
      **vmp_map_kwargs, params_flow_init=params_flow_init)(
          eta)


def elbo_estimate_vmap(
    params_vmap_tuple: Tuple[hk.Params],
    batch: Batch,
    prng_key: PRNGKey,
    num_samples_eta: int,
    num_samples_flow: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    vmp_map_name: str,
    vmp_map_kwargs: Dict[str, Any],
    params_flow_init_list: List[hk.Params],
    eta_name: str,
    eta_dim: int,
    eta_sampling_a: float,
    eta_sampling_b: float,
) -> Dict[str, Array]:
  """Estimate ELBO.

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  # Sample eta values
  prng_key, key_eta = jax.random.split(prng_key)
  etas = jax.random.beta(
      key=key_eta,
      a=eta_sampling_a,
      b=eta_sampling_b,
      shape=(num_samples_eta, eta_dim),
  )
  # # Clip eta so that the variance of the prior p(beta|tau is not too large)
  # etas = jnp.clip(etas, 1e-4)

  smi_eta_vmap = {eta_name: etas}

  params_flow_tuple = [
      vmp_map.apply(
          params_vmap,
          eta=etas,
          vmp_map_name=vmp_map_name,
          vmp_map_kwargs=vmp_map_kwargs,
          params_flow_init=params_flow_init,
      ) for params_vmap, params_flow_init in zip(params_vmap_tuple,
                                                 params_flow_init_list)
  ]
  # globals().update(vmp_map_kwargs)

  # Use same key for every eta
  # (same key implies same samples from the base distribution)
  prng_key, key_elbo = jax.random.split(prng_key)

  # Compute ELBO.
  elbo_dict = jax.vmap(lambda params_flow_tuple_i, smi_eta_i: elbo_estimate(
      params_tuple=params_flow_tuple_i,
      batch=batch,
      prng_key=key_elbo,
      num_samples=num_samples_flow,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      smi_eta=smi_eta_i,
  ))(params_flow_tuple, smi_eta_vmap)

  return elbo_dict


def loss(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  # Compute ELBO.
  elbo_dict = elbo_estimate_vmap(
      params_vmap_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2']))

  return loss_avg


def plot_vmp_map(
    state: TrainState,
    vmp_map_name: str,
    vmp_map_kwargs: Mapping[str, Any],
    params_flow_init: hk.Params,
    lambda_idx: Union[int, List[int]],
    eta_grid: Array,
    eta_grid_x_y_idx: Tuple[int, int],
    constant_lambda_ignore_plot: bool,
):
  """Visualize VMP map."""

  assert eta_grid.ndim == 3

  num_groups, *grid_shape = eta_grid.shape
  grid_shape = tuple(grid_shape)

  params_flow_grid = vmp_map.apply(
      state.params,
      eta=eta_grid.reshape(num_groups, -1).T,
      vmp_map_name=vmp_map_name,
      vmp_map_kwargs=vmp_map_kwargs,
      params_flow_init=params_flow_init,
  )
  # # All variational parameters (too much memory)
  # lambda_all = jnp.concatenate([
  #     x.reshape((eta_grid_len + 1)**2, -1)
  #     for x in jax.tree_util.tree_leaves(params_flow_grid)
  # ],
  #                              axis=-1)
  lambda_all = jax.tree_util.tree_leaves(params_flow_grid)[0].reshape(
      math.prod(grid_shape), -1)

  # Ignore flat functions of eta
  if constant_lambda_ignore_plot:
    lambda_all = lambda_all[:,
                            jnp.where(
                                jnp.square(lambda_all - lambda_all[[0], :]).sum(
                                    axis=0) > 0.)[0]]

  fig, axs = plt.subplots(
      nrows=1,
      ncols=len(lambda_idx),
      figsize=(4 * (len(lambda_idx)), 3),
      subplot_kw={"projection": "3d"},
  )
  if not lambda_all.shape[1] > 0:
    return fig, axs

  lambda_all = lambda_all.reshape(grid_shape + (-1,))

  if len(lambda_idx) == 1:
    axs = [axs]
  # Plot the surface.
  for i, idx_i in enumerate(lambda_idx):
    # i=0; idx_i=idx[i]
    axs[i].plot_surface(
        eta_grid[eta_grid_x_y_idx[0]],
        eta_grid[eta_grid_x_y_idx[1]],
        lambda_all[:, :, idx_i],
        cmap=matplotlib.cm.inferno,
        # linewidth=0,
        # antialiased=False,
    )
    axs[i].view_init(30, 225)
    axs[i].set_xlabel(f'eta_{eta_grid_x_y_idx[0]}')
    axs[i].set_ylabel(f'eta_{eta_grid_x_y_idx[1]}')
    axs[i].set_title(f'lambda_{lambda_idx[i]}')

  fig.tight_layout()

  return fig, axs


def sample_for_eval(state_list, prng_key, etas, config, params_flow_init_list):
  """Auxiliary function to get posterior samples for computing ELPD."""
  # Obtain variational parameters for each eta value
  params_flow_tuple = [
      vmp_map.apply(
          state.params,
          eta=etas,
          vmp_map_name=config.vmp_map_name,
          vmp_map_kwargs=config.vmp_map_kwargs,
          params_flow_init=params_flow_init,
      ) for state, params_flow_init in zip(state_list, params_flow_init_list)
  ]

  # Sample from posteriors for each eta value
  # (use a single key to use same samples from the base distribution)
  q_distr_out = jax.vmap(lambda params_flow_tuple_i: sample_all_flows(
      params_tuple=params_flow_tuple_i,
      prng_key=prng_key,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_eval,),
  ))(
      params_flow_tuple)

  return q_distr_out


def plot_elpd_surface(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    y_new: Array,
    config: ConfigDict,
    eta_grid: Array,
    eta_grid_x_y_idx: Tuple[int, int],
    params_flow_init_list: List[hk.Params],
):
  """Visualize ELPD surface as function of eta."""

  assert eta_grid.ndim == 3

  num_groups, *grid_shape = eta_grid.shape

  # TODO: vmap implementation produces RuntimeError: RESOURCE_EXHAUSTED
  lpd_pointwise_all_eta = []
  elpd_mc_pointwise_all_eta = []
  elpd_waic_pointwise_all_eta = []
  for eta_i in (eta_grid.reshape(num_groups, -1).T):
    # eta_i = (eta_grid.reshape(num_groups, -1).T)[0]

    eta_i = jnp.expand_dims(eta_i, axis=0)
    q_distr_out_i = sample_for_eval(
        state_list=state_list,
        prng_key=prng_key,  # same key to reduce variance of posterior along eta
        etas=eta_i,
        config=config,
        params_flow_init_list=params_flow_init_list,
    )
    elpd_dict_i = compute_elpd_jit(
        posterior_sample_dict=jax.tree_map(lambda x: x[0],
                                           q_distr_out_i['posterior_sample']),
        batch=batch,
        y_new=y_new,
    )
    # elpd_dict_i = jax.vmap(
    #     lambda posterior_sample: compute_elpd_jit(
    #         posterior_sample_dict=posterior_sample,
    #         batch=batch,
    #         y_new=y_new,
    #     ))(
    #         posterior_sample_i)
    lpd_pointwise_all_eta.append(elpd_dict_i['lpd_pointwise'])
    elpd_mc_pointwise_all_eta.append(elpd_dict_i['elpd_mc_pointwise'])
    elpd_waic_pointwise_all_eta.append(elpd_dict_i['elpd_waic_pointwise'])

  lpd_pointwise_all_eta = jnp.stack(lpd_pointwise_all_eta, axis=0)
  elpd_mc_pointwise_all_eta = jnp.stack(elpd_mc_pointwise_all_eta, axis=0)
  elpd_waic_pointwise_all_eta = jnp.stack(elpd_waic_pointwise_all_eta, axis=0)

  # Add pointwise elpd and lpd across observations
  lpd_all_eta = lpd_pointwise_all_eta.sum(axis=-1).reshape(grid_shape)
  elpd_mc_all_eta = elpd_mc_pointwise_all_eta.sum(axis=-1).reshape(grid_shape)
  elpd_waic_all_eta = elpd_waic_pointwise_all_eta.sum(
      axis=-1).reshape(grid_shape)

  # Plot the ELPD surface.
  fig, axs = plt.subplots(
      nrows=1, ncols=3, figsize=(4 * 3, 3), subplot_kw={"projection": "3d"})
  # i=0; idx_i=idx[i]
  for i, metric in enumerate([lpd_all_eta, elpd_mc_all_eta, elpd_waic_all_eta]):
    axs[i].plot_surface(
        eta_grid[eta_grid_x_y_idx[0]],
        eta_grid[eta_grid_x_y_idx[1]],
        -metric,
        cmap=matplotlib.cm.inferno,
        # linewidth=0,
        # antialiased=False,
    )
    axs[i].view_init(30, 225)
    axs[i].set_xlabel(f'eta_{eta_grid_x_y_idx[0]}')
    axs[i].set_ylabel(f'eta_{eta_grid_x_y_idx[1]}')
    axs[i].set_title(['- LPD', '- ELPD', '- ELPD WAIC'][i])

  fig.tight_layout()

  return fig, axs


def log_images(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    config: ConfigDict,
    state_name_list: List[str],
    show_vmp_map: bool,
    eta_grid_len: int,
    params_flow_init_list: List[hk.Params],
    show_elpd: bool,
    y_new: Array,
    summary_writer: Optional[SummaryWriter],
    workdir_png: Optional[str],
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  eta_plot = jnp.array(config.eta_plot)

  assert eta_plot.ndim == 2

  # Produce flow parameters as a function of eta
  params_flow_tuple = [
      vmp_map.apply(
          state.params,
          eta=eta_plot,
          vmp_map_name=config.vmp_map_name,
          vmp_map_kwargs=config.vmp_map_kwargs,
          params_flow_init=params_flow_init,
      ) for state, params_flow_init in zip(state_list, params_flow_init_list)
  ]

  # Use same key for every eta
  # (same key implies same samples from the base distribution)
  key_flow = next(prng_seq)

  # Sample from flow
  q_distr_out = jax.vmap(lambda params_flow_tuple: sample_all_flows(
      params_tuple=params_flow_tuple,
      prng_key=key_flow,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_plot,),
  ))(
      params_flow_tuple)

  # Plot posterior samples
  for i in range(eta_plot.shape[0]):
    posterior_sample_dict_i = jax.tree_util.tree_map(
        lambda x: x[i],  # pylint: disable=cell-var-from-loop
        q_distr_out['posterior_sample'])
    plot.posterior_samples(
        posterior_sample_dict=posterior_sample_dict_i,
        step=state_list[0].step,
        summary_writer=summary_writer,
        suffix=config.suffix_eta_plot[i],
        workdir_png=workdir_png,
    )

  ### Visualize meta-posterior map along the Eta space ###

  # Define elements to grate grid of eta values
  eta_base = np.array([0., 0.] + [1. for _ in range(config.num_groups - 2)])
  eta_grid_base = np.tile(eta_base, [eta_grid_len + 1, eta_grid_len + 1, 1]).T
  eta_grid_mini = np.stack(
      np.meshgrid(
          np.linspace(0, 1, eta_grid_len + 1),
          np.linspace(0, 1, eta_grid_len + 1)),
      axis=0)

  if show_vmp_map:

    ### VMP-map ###
    images = []

    for state_i, state_name_i, params_flow_init_i in zip(
        state_list, state_name_list, params_flow_init_list):
      # Vary eta_0 and eta_1
      plot_name = 'rnd_eff_vmp_map_eta_0_1_' + state_name_i

      eta_grid = eta_grid_base.copy()
      eta_grid_x_y_idx = [0, 1]
      eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

      fig, _ = plot_vmp_map(
          state=state_i,
          vmp_map_name=config.vmp_map_name,
          vmp_map_kwargs=config.vmp_map_kwargs,
          params_flow_init=params_flow_init_i,
          lambda_idx=np.array(config.lambda_idx_plot),
          eta_grid=eta_grid,
          eta_grid_x_y_idx=eta_grid_x_y_idx,
          constant_lambda_ignore_plot=config.constant_lambda_ignore_plot,
      )
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

      # Vary eta_0 and eta_2
      plot_name = 'rnd_eff_vmp_map_eta_0_2_' + state_name_i

      eta_grid = eta_grid_base.copy()
      eta_grid_x_y_idx = [0, 2]
      eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

      fig, _ = plot_vmp_map(
          state=state_i,
          vmp_map_name=config.vmp_map_name,
          vmp_map_kwargs=config.vmp_map_kwargs,
          params_flow_init=params_flow_init_i,
          lambda_idx=np.array(config.lambda_idx_plot),
          eta_grid=eta_grid,
          eta_grid_x_y_idx=eta_grid_x_y_idx,
          constant_lambda_ignore_plot=config.constant_lambda_ignore_plot,
      )
      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    # Logging VMP-map plots
    if summary_writer:
      plot_name = 'rnd_eff_vmp_map'
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=state_list[0].step,
      )

  ### ELPD ###
  if show_elpd:
    # varying eta across eta_1 and eta_2, rest eta's fixed to 1
    images = []
    prng_key_elpd = next(prng_seq)

    # Vary eta_0 and eta_1
    plot_name = 'elpd_surface_eta_0_1'

    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [0, 1]
    eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

    fig, _ = plot_elpd_surface(
        state_list=state_list,
        batch=batch,
        prng_key=prng_key_elpd,
        y_new=y_new,
        config=config,
        eta_grid=eta_grid,
        eta_grid_x_y_idx=eta_grid_x_y_idx,
        params_flow_init_list=params_flow_init_list,
    )
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    # Vary eta_0 and eta_2
    plot_name = 'elpd_surface_eta_0_2'

    eta_grid = eta_grid_base.copy()
    eta_grid_x_y_idx = [0, 2]
    eta_grid[eta_grid_x_y_idx, :, :] = eta_grid_mini

    fig, _ = plot_elpd_surface(
        state_list=state_list,
        batch=batch,
        prng_key=prng_key_elpd,
        y_new=y_new,
        config=config,
        eta_grid=eta_grid,
        eta_grid_x_y_idx=eta_grid_x_y_idx,
        params_flow_init_list=params_flow_init_list,
    )
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      images.append(plot_to_image(fig))

    # Vary eta misspecified and eta well-specified
    show_elpd_good_bad = False
    if show_elpd_good_bad:
      plot_name = 'elpd_surface_eta_good_bad'

      eta_bad, eta_good = np.split(eta_grid_mini, 2, axis=0)
      eta_grid = np.concatenate(
          [
              np.tile(eta_bad, [2, 1, 1]),
              np.tile(eta_good, [config.num_groups - 2, 1, 1])
          ],
          axis=0,
      )

      fig, axs = plot_elpd_surface(
          state_list=state_list,
          batch=batch,
          prng_key=prng_key_elpd,
          y_new=y_new,
          config=config,
          eta_grid=eta_grid,
          eta_grid_x_y_idx=[0, 2],
          params_flow_init_list=params_flow_init_list,
      )
      for ax in axs:
        ax.set_xlabel('eta_bad')
        ax.set_ylabel('eta_good')

      if workdir_png:
        fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
      if summary_writer:
        images.append(plot_to_image(fig))

    if summary_writer:
      plot_name = 'rnd_eff_elpd_surface'
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=state_list[0].step,
      )


def train_and_evaluate(config: ConfigDict, workdir: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  # Initialize random key sequence
  prng_seq = hk.PRNGSequence(config.seed)

  # Full dataset used everytime
  # Small data, no need to batch
  train_ds = get_dataset(
      num_obs_groups=config.num_obs_groups,
      loc_groups=config.loc_groups,
      scale_groups=config.scale_groups,
      prng_key=next(prng_seq),
  )

  # Dictionary of true generative parameters
  # Used in ELPD evaluation
  true_params_dict = {}
  true_params_dict['beta'] = jnp.array(config.loc_groups)
  true_params_dict['sigma'] = jnp.array(config.scale_groups)
  # Generate new observations from the true generative model
  y_new = distrax.Normal(
      loc=true_params_dict['beta'][train_ds['group']],
      scale=true_params_dict['sigma'][train_ds['group']],
  ).sample(
      seed=next(prng_seq), sample_shape=(config.num_samples_eval,))

  # num_groups is also a parameter of the flow,
  # as it define its dimension
  config.flow_kwargs.num_groups = config.num_groups
  config.flow_kwargs.is_smi = True

  # smi_eta = {'groups':jnp.ones((1,2))}
  # elbo_estimate(config.params_flow, next(prng_seq), train_ds, smi_eta)

  state_name_list = ['sigma', 'beta_tau', 'beta_tau_aux']

  # Get examples of the output tree to be produced by the meta functions
  params_flow_init_list = []

  params_flow_init_list.append(
      hk.transform(q_distr_sigma).init(
          next(prng_seq),
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          sample_shape=(config.num_samples_elbo,),
      ))

  # Get an initial sample of sigma
  # (used below to initialize beta and tau)
  sigma_base_sample_init = hk.transform(q_distr_sigma).apply(
      params_flow_init_list[0],
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_elbo,),
  )['sigma_base_sample']

  params_flow_init_list.append(
      hk.transform(q_distr_beta_tau).init(
          next(prng_seq),
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          sigma_base_sample=sigma_base_sample_init,
          is_aux=False,
      ))

  params_flow_init_list.append(
      hk.transform(q_distr_beta_tau).init(
          next(prng_seq),
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          sigma_base_sample=sigma_base_sample_init,
          is_aux=True,
      ))

  ### Set Variational Meta-Posterior Map ###

  ### Initialize States ###
  # Here we use three different states defining three separate flow models:
  #   -sigma
  #   -beta and tau
  #   -auxiliary beta and tau

  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')

  state_list = [
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
          forward_fn=vmp_map,
          forward_fn_kwargs={
              'eta': jnp.ones((config.num_samples_eta, config.num_groups)),
              'vmp_map_name': config.vmp_map_name,
              'vmp_map_kwargs': config.vmp_map_kwargs,
              'params_flow_init': params_flow_init_i,
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ) for state_name_i, params_flow_init_i in zip(state_name_list,
                                                    params_flow_init_list)
  ]

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)
  if jax.process_index() == 0 and state_list[0].step < config.training_steps:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))
  else:
    summary_writer = None

  # Print a useful summary of the execution of the VHP-map architecture.
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda state_i, params_flow_init_i: vmp_map.apply(
          state_i.params,
          eta=jnp.ones((config.num_samples_eta, config.num_groups)),
          vmp_map_name=config.vmp_map_name,
          vmp_map_kwargs=config.vmp_map_kwargs,
          params_flow_init=params_flow_init_i,
      ),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_list[0], params_flow_init_list[0])
  logging.info('VMP-MAP SIGMA:')
  for line in summary.split("\n"):
    logging.info(line)

  summary = tabulate_fn_(state_list[1], params_flow_init_list[1])
  logging.info('VMP-MAP BETA TAU:')
  for line in summary.split("\n"):
    logging.info(line)

  ### Training VMP map ###

  update_states_jit = lambda state_list, batch, prng_key: update_states(
      state_list=state_list,
      batch=batch,
      prng_key=prng_key,
      optimizer=make_optimizer(**config.optim_kwargs),
      loss_fn=loss,
      loss_fn_kwargs={
          'num_samples_eta': config.num_samples_eta,
          'num_samples_flow': config.num_samples_elbo,
          'flow_name': config.flow_name,
          'flow_kwargs': config.flow_kwargs,
          'vmp_map_name': config.vmp_map_name,
          'vmp_map_kwargs': config.vmp_map_kwargs,
          'params_flow_init_list': params_flow_init_list,
          'eta_name': 'groups',
          'eta_dim': config.num_groups,
          'eta_sampling_a': config.eta_sampling_a,
          'eta_sampling_b': config.eta_sampling_b,
      },
  )
  # globals().update(loss_fn_kwargs)
  update_states_jit = jax.jit(update_states_jit)

  # Compute average ELBO in two stages for evaluation
  def elbo_validation_jit(state_list, batch, prng_key, eta_sampling_a,
                          eta_sampling_b):
    prng_seq = hk.PRNGSequence(prng_key)

    etas_avg_elbo = jax.random.beta(
        key=next(prng_seq),
        a=eta_sampling_a,
        b=eta_sampling_b,
        shape=(config.num_samples_eta, config.num_groups),
    )
    params_flow_tuple = [
        vmp_map.apply(
            state.params,
            eta=etas_avg_elbo,
            vmp_map_name=config.vmp_map_name,
            vmp_map_kwargs=config.vmp_map_kwargs,
            params_flow_init=params_flow_init,
        ) for state, params_flow_init in zip(state_list, params_flow_init_list)
    ]
    elbo_dict_val = jax.vmap(
        lambda params_flow_tuple_i, smi_eta_i: elbo_estimate(
            params_tuple=params_flow_tuple_i,
            batch=batch,
            prng_key=next(prng_seq),
            num_samples=config.num_samples_eval,
            flow_name=config.flow_name,
            flow_kwargs=config.flow_kwargs,
            smi_eta=smi_eta_i,
        ))(params_flow_tuple, {
            'groups': etas_avg_elbo
        })
    return elbo_dict_val

  elbo_validation_jit = jax.jit(elbo_validation_jit)

  save_after_training = False
  if state_list[0].step < config.training_steps:
    save_after_training = True
    logging.info('Training Variational Meta-Posterior (VMP-map)...')
    # Reset random key sequence
    prng_seq = hk.PRNGSequence(config.seed)

  while state_list[0].step < config.training_steps:

    # Plots to monitor training
    if (config.log_img_steps
        is not None) and ((state_list[0].step in [0, 1]) or
                          (state_list[0].step % config.log_img_steps == 0)):
      # print("Logging images...\n")
      log_images(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
          config=config,
          state_name_list=state_name_list,
          show_vmp_map=True,
          eta_grid_len=10,
          params_flow_init_list=params_flow_init_list,
          show_elpd=False,
          y_new=y_new,
          summary_writer=summary_writer,
          workdir_png=workdir,
      )
      plt.close()

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

    if state_list[0].step % config.eval_steps == 0:
      elbo_avg_dict = elbo_validation_jit(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
          eta_sampling_a=1.,
          eta_sampling_b=1.,
      )
      for k, v in elbo_avg_dict.items():
        summary_writer.scalar(
            tag=f'elbo_{k}',
            value=v.mean(),
            step=state_list[0].step,
        )

    if state_list[0].step == 2:
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

    # Metrics for evaluation
    if state_list[0].step % config.eval_steps == 0:

      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

      # ELPD values on eta_plot values
      q_distr_out_eta_plot = sample_for_eval(
          state_list=state_list,
          prng_key=next(prng_seq),
          etas=jnp.array(config.eta_plot),
          config=config,
          params_flow_init_list=params_flow_init_list,
      )
      elpd_dict = jax.vmap(lambda posterior_sample_i: compute_elpd_jit(
          posterior_sample_dict=posterior_sample_i,
          batch=train_ds,
          y_new=y_new,
      ))(
          q_distr_out_eta_plot['posterior_sample'])

      # Pointwise elpd and lpd across observations to tensorboard
      for i, suffix in enumerate(config.suffix_eta_plot):
        # i=0; suffix=config.suffix_eta_plot[i]
        summary_writer.scalar(
            tag=f'lpd_{suffix}',
            value=elpd_dict['lpd_pointwise'][i].sum(),
            step=state_list[0].step,
        )
        summary_writer.scalar(
            tag=f'elpd_mc_{suffix}',
            value=elpd_dict['elpd_mc_pointwise'][i].sum(),
            step=state_list[0].step,
        )
        summary_writer.scalar(
            tag=f'elpd_waic_{suffix}',
            value=elpd_dict['elpd_waic_pointwise'][i].sum(),
            step=state_list[0].step,
        )

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
  if save_after_training:
    for state_i, state_name_i in zip(state_list, state_name_list):
      save_checkpoint(
          state=state_i,
          checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
          keep=config.checkpoints_keep,
      )

  # Last plot of posteriors
  log_images(
      state_list=state_list,
      batch=train_ds,
      prng_key=jax.random.PRNGKey(config.seed),
      config=config,
      state_name_list=state_name_list,
      show_vmp_map=True,
      eta_grid_len=10,
      params_flow_init_list=params_flow_init_list,
      show_elpd=True,
      y_new=y_new,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )
  plt.close()

  return state_list
