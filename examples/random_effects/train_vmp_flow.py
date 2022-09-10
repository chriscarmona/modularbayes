"""A simple example of variational SMI on the Random Effects model."""
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
import distrax

import flows
import log_prob_fun
import plot

from train_flow import compute_elpd, get_dataset, make_optimizer

from modularbayes._src.utils.training import TrainState
from modularbayes import (plot_to_image, normalize_images, flatten_dict,
                          initial_state_ckpt, update_state, update_states,
                          save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict, List,
                                      Optional, PRNGKey, SmiEta, SummaryWriter,
                                      Tuple)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)

compute_elpd_jit = jax.jit(compute_elpd)


def make_optimizer_eta(learning_rate: float) -> optax.GradientTransformation:
  optimizer = optax.adabelief(learning_rate=learning_rate)
  return optimizer


def q_distr_sigma(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    eta: Array,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  # Define normalizing flows
  q_distr = getattr(flows, flow_name + '_sigma')(**flow_kwargs)

  num_samples = eta.shape[0]

  # Sample from flow
  (sigma_sample, sigma_log_prob_posterior,
   sigma_base_sample) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=(num_samples,),
       context=[eta, None],
   )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows.split_flow_sigma(
          samples=sigma_sample,
          **flow_kwargs,
      ))

  # sample from base distribution that generated sigma
  q_distr_out['sigma_base_sample'] = sigma_base_sample

  # log P(sigma)
  q_distr_out['sigma_log_prob'] = sigma_log_prob_posterior

  return q_distr_out


def q_distr_beta_tau(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    sigma_base_sample: Array,
    eta: Array,
    is_aux: bool,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  num_samples = sigma_base_sample.shape[0]

  # Define normalizing flows
  q_distr = getattr(flows, flow_name + '_beta_tau')(**flow_kwargs)

  # Sample from flow
  (beta_tau_sample, beta_tau_log_prob_posterior) = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
      context=[eta, sigma_base_sample],
  )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows.split_flow_beta_tau(
          samples=beta_tau_sample,
          is_aux=is_aux,
          **flow_kwargs,
      ))

  # log P(beta,tau|sigma)
  q_distr_out['beta_tau_' + ('aux_' if is_aux else '') +
              'log_prob'] = beta_tau_log_prob_posterior

  return q_distr_out


def sample_all_flows(
    params_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    smi_eta: SmiEta,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  prng_seq = hk.PRNGSequence(prng_key)

  # sigma
  q_distr_out = hk.transform(q_distr_sigma).apply(
      params_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      eta=smi_eta['groups'],
  )

  # beta and tau
  q_distr_out_beta_tau = hk.transform(q_distr_beta_tau).apply(
      params_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sigma_base_sample=q_distr_out['sigma_base_sample'],
      eta=smi_eta['groups'],
      is_aux=False,
  )
  q_distr_out['posterior_sample'].update(
      q_distr_out_beta_tau['posterior_sample'])
  q_distr_out['beta_tau_log_prob'] = q_distr_out_beta_tau['beta_tau_log_prob']

  q_distr_out_beta_tau_aux = hk.transform(q_distr_beta_tau).apply(
      params_tuple[2],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sigma_base_sample=q_distr_out['sigma_base_sample'],
      eta=smi_eta['groups'],
      is_aux=True,
  )
  q_distr_out['posterior_sample'].update(
      q_distr_out_beta_tau_aux['posterior_sample'])
  q_distr_out['beta_tau_aux_log_prob'] = q_distr_out_beta_tau_aux[
      'beta_tau_aux_log_prob']

  return q_distr_out


def elbo_estimate_along_eta(
    params_tuple: Tuple[hk.Params],
    batch: Batch,
    prng_key: PRNGKey,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    eta_sampling_a: float,
    eta_sampling_b: float,
) -> Dict[str, Array]:
  """Estimate ELBO

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  prng_seq = hk.PRNGSequence(prng_key)

  # Sample eta values
  etas_elbo = jax.random.beta(
      key=next(prng_seq),
      a=eta_sampling_a,
      b=eta_sampling_b,
      shape=(num_samples, flow_kwargs.num_groups),
  )
  smi_eta_elbo = {'groups': etas_elbo}

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=params_tuple,
      prng_key=prng_key,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      smi_eta=smi_eta_elbo,
  )

  shared_params_names = [
      'sigma',
  ]
  refit_params_names = [
      'beta',
      'tau',
  ]

  # ELBO stage 1: Power posterior
  posterior_sample_dict_stg1 = {}
  for key in shared_params_names:
    posterior_sample_dict_stg1[key] = q_distr_out['posterior_sample'][key]
  for key in refit_params_names:
    posterior_sample_dict_stg1[key] = q_distr_out['posterior_sample'][key +
                                                                      '_aux']

  log_prob_joint_stg1 = jax.vmap(
      lambda posterior_sample_i, smi_eta_i: log_prob_fun.log_prob_joint(
          batch=batch,
          posterior_sample_dict=posterior_sample_i,
          smi_eta=smi_eta_i,
      ))(
          jax.tree_map(lambda x: jnp.expand_dims(x, 1),
                       posterior_sample_dict_stg1),
          smi_eta_elbo,
      )

  log_q_stg1 = (
      q_distr_out['sigma_log_prob'] + q_distr_out['beta_tau_aux_log_prob'])

  elbo_stg1 = log_prob_joint_stg1.reshape(-1) - log_q_stg1

  # ELBO stage 2: Refit theta
  posterior_sample_dict_stg2 = {}
  for key in shared_params_names:
    posterior_sample_dict_stg2[key] = jax.lax.stop_gradient(
        q_distr_out['posterior_sample'][key])
  for key in refit_params_names:
    posterior_sample_dict_stg2[key] = q_distr_out['posterior_sample'][key]

  log_prob_joint_stg2 = jax.vmap(
      lambda posterior_sample_i: log_prob_fun.log_prob_joint(
          batch=batch,
          posterior_sample_dict=posterior_sample_i,
          smi_eta=None,
      ))(
          jax.tree_map(lambda x: jnp.expand_dims(x, 1),
                       posterior_sample_dict_stg2))

  log_q_stg2 = (
      jax.lax.stop_gradient(q_distr_out['sigma_log_prob']) +
      q_distr_out['beta_tau_log_prob'])

  elbo_stg2 = log_prob_joint_stg2.reshape(-1) - log_q_stg2

  elbo_dict = {'stage_1': elbo_stg1, 'stage_2': elbo_stg2}

  return elbo_dict


def loss(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  ### Compute ELBO ###
  elbo_dict = elbo_estimate_along_eta(
      params_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2']))

  return loss_avg


def elpd_estimate_one_eta(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    num_samples: int,
    eta: Array,
    y_new: Optional[Array] = None,
):
  q_distr_out_i = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      prng_key=prng_key,  # same key to reduce variance of posterior along eta
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      smi_eta={
          'groups': jnp.broadcast_to(eta, (num_samples, flow_kwargs.num_groups))
      },
  )
  elpd_dict = compute_elpd_jit(
      posterior_sample_dict=q_distr_out_i['posterior_sample'],
      batch=batch,
      y_new=y_new,
  )
  return elpd_dict


def maximize_elpd(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    config: ConfigDict,
    num_samples_elpd: int,
) -> Array:

  # Jit the computation of elpd
  elpd_waic_one_eta_jit = lambda state_list_i, prng_key_i, eta_i: elpd_estimate_one_eta(
      state_list=state_list_i,
      batch=batch,
      prng_key=prng_key_i,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      num_samples=num_samples_elpd,
      eta=eta_i,
      y_new=None,
  )['elpd_waic_pointwise'].sum(axis=-1)
  elpd_waic_one_eta_jit = jax.jit(elpd_waic_one_eta_jit)

  # Define lambda function only depending on eta
  fun_objective = lambda eta_i: elpd_waic_one_eta_jit(state_list, prng_key,
                                                      eta_i)

  return None


def plot_elpd_surface(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    y_new: Array,
    config: ConfigDict,
    eta_grid: Array,
    eta_grid_x_y_idx: Tuple[int, int],
    use_vmap: bool = True,
):
  """Visualize ELPD surface as function of eta."""

  assert eta_grid.ndim == 3

  num_groups, *grid_shape = eta_grid.shape

  # TODO: vmap implementation produces RuntimeError: RESOURCE_EXHAUSTED
  lpd_pointwise_all_eta = []
  elpd_mc_pointwise_all_eta = []
  elpd_waic_pointwise_all_eta = []

  if use_vmap:
    # Faster option: using vmap
    # Sometimes fail due to RuntimeError: RESOURCE_EXHAUSTED
    elpd_dict_all = jax.vmap(lambda eta_i: elpd_estimate_one_eta(
        state_list=state_list,
        batch=batch,
        prng_key=prng_key,
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        num_samples=config.num_samples_elpd,
        eta=eta_i,
        y_new=y_new,
    ))(
        eta_grid.reshape(num_groups, -1).T)

    lpd_pointwise_all_eta = elpd_dict_all['lpd_pointwise']
    elpd_mc_pointwise_all_eta = elpd_dict_all['elpd_mc_pointwise']
    elpd_waic_pointwise_all_eta = elpd_dict_all['elpd_waic_pointwise']
  else:
    # Slower option: for loop
    # Takes longer to compute
    for eta_i in (eta_grid.reshape(num_groups, -1).T):
      # eta_i = (eta_grid.reshape(num_groups, -1).T)[0]

      elpd_dict_i = elpd_estimate_one_eta(
          state_list=state_list,
          batch=batch,
          prng_key=prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          num_samples=config.num_samples_elpd,
          eta=eta_i,
          y_new=y_new,
      )
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
    show_elpd: bool,
    eta_grid_len: int,
    y_new: Optional[Array],
    summary_writer: Optional[SummaryWriter],
    workdir_png: Optional[str],
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  eta_plot = jnp.array(config.eta_plot)

  assert eta_plot.ndim == 2

  # Plot posterior samples
  key_flow = next(prng_seq)
  for i in range(eta_plot.shape[0]):
    # Sample from flow
    q_distr_out = sample_all_flows(
        params_tuple=[state.params for state in state_list],
        prng_key=key_flow,
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        smi_eta={
            'groups':
                jnp.broadcast_to(eta_plot[[i], :], (config.num_samples_plot,) +
                                 eta_plot.shape[1:])
        },
    )

    plot.posterior_samples(
        posterior_sample_dict=q_distr_out['posterior_sample'],
        step=state_list[0].step,
        summary_writer=summary_writer,
        suffix=config.suffix_eta_plot[i],
        workdir_png=workdir_png,
    )

  ### ELPD ###

  # Define elements to grate grid of eta values
  eta_base = np.array([0., 0.] + [1. for _ in range(config.num_groups - 2)])
  eta_grid_base = np.tile(eta_base, [eta_grid_len + 1, eta_grid_len + 1, 1]).T
  eta_grid_mini = np.stack(
      np.meshgrid(
          np.linspace(0., 1., eta_grid_len + 1),
          np.linspace(0., 1., eta_grid_len + 1)),
      axis=0)

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
    )
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

  # Initialize random keys
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
  # Also is_smi modifies the dimension of the flow, due to the duplicated params
  config.flow_kwargs.is_smi = True

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)
  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))
  else:
    summary_writer = None

  ### Initialize States ###
  # Here we use three different states defining three separate flow models:
  #   -sigma
  #   -beta and tau
  #   -auxiliary beta and tau

  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')
  state_list = []
  state_name_list = ['sigma', 'beta_tau', 'beta_tau_aux']

  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[0]}',
          forward_fn=hk.transform(q_distr_sigma),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'eta': jnp.ones((config.num_samples_elbo, config.num_groups))
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))
  # globals().update(forward_fn_kwargs)

  # Get an initial sample of sigma
  # (used below to initialize beta and tau)
  sigma_base_sample_init = hk.transform(q_distr_sigma).apply(
      state_list[0].params,
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      eta=jnp.ones((config.num_samples_elbo, config.num_groups)),
  )['sigma_base_sample']

  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[1]}',
          forward_fn=hk.transform(q_distr_beta_tau),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'sigma_base_sample': sigma_base_sample_init,
              'eta': jnp.ones((config.num_samples_elbo, config.num_groups)),
              'is_aux': False,
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[2]}',
          forward_fn=hk.transform(q_distr_beta_tau),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'sigma_base_sample': sigma_base_sample_init,
              'eta': jnp.ones((config.num_samples_elbo, config.num_groups)),
              'is_aux': True,
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # Print a useful summary of the execution of the flow architecture.
  logging.info('FLOW SIGMA:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: hk.transform(q_distr_sigma).apply(
          params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          eta=jnp.ones((config.num_samples_elbo, config.num_groups)),
      ),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_list[0].params, next(prng_seq))
  for line in summary.split("\n"):
    logging.info(line)

  logging.info('FLOW BETA TAU:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: hk.transform(q_distr_beta_tau).apply(
          params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          sigma_base_sample=sigma_base_sample_init,
          eta=jnp.ones((config.num_samples_elbo, config.num_groups)),
          is_aux=False,
      ),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_list[1].params, next(prng_seq))
  for line in summary.split("\n"):
    logging.info(line)

  # Jit function to update training states
  update_states_jit = lambda state_list, batch, prng_key: update_states(
      state_list=state_list,
      batch=batch,
      prng_key=prng_key,
      optimizer=make_optimizer(**config.optim_kwargs),
      loss_fn=loss,
      loss_fn_kwargs={
          'num_samples': config.num_samples_elbo,
          'flow_name': config.flow_name,
          'flow_kwargs': config.flow_kwargs,
          'eta_sampling_a': config.eta_sampling_a,
          'eta_sampling_b': config.eta_sampling_b,
      },
  )
  update_states_jit = jax.jit(update_states_jit)

  elbo_validation_jit = lambda state_list, batch, prng_key: elbo_estimate_along_eta(
      params_tuple=[state.params for state in state_list],
      batch=batch,
      prng_key=prng_key,
      num_samples=config.num_samples_eval,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      eta_sampling_a=1.,
      eta_sampling_b=1.,
  )
  elbo_validation_jit = jax.jit(elbo_validation_jit)

  # elpd waic as a loss function to optimize eta
  loss_neg_elpd = lambda eta, batch, prng_key, state_list_vmp: -elpd_estimate_one_eta(
      state_list=state_list_vmp,
      batch=batch,
      prng_key=prng_key,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      num_samples=config.num_samples_elpd,
      eta=eta,
      y_new=None,
  )['elpd_waic_pointwise'].sum(axis=-1)
  loss_neg_elpd = jax.jit(loss_neg_elpd)
  # loss_neg_elpd(
  #     eta=jnp.ones((config.num_groups)),
  #     batch=train_ds,
  #     prng_key=next(prng_seq),
  #     state_list_vmp=state_list,
  # )

  # Jit optimization of eta
  update_eta_star_state = lambda eta_star_state, batch, prng_key: update_state(
      state=eta_star_state,
      batch=batch,
      prng_key=prng_key,
      optimizer=make_optimizer_eta(**config.optim_kwargs_eta),
      loss_fn=loss_neg_elpd,
      loss_fn_kwargs={
          'state_list_vmp': state_list,
      },
  )
  update_eta_star_state = jax.jit(update_eta_star_state)

  save_after_training = False
  if state_list[0].step < config.training_steps:
    save_after_training = True
    logging.info('Training Variational Meta-Posterior (VMP-flow)...')
    # Reset random key sequence
    prng_seq = hk.PRNGSequence(config.seed)

  while state_list[0].step < config.training_steps:

    # Plots to monitor during training
    if ((state_list[0].step == 0) or
        (state_list[0].step % config.log_img_steps == 0)):
      # print("Logging images...\n")
      log_images(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
          config=config,
          show_elpd=False,
          eta_grid_len=10,
          y_new=y_new,
          summary_writer=summary_writer,
          workdir_png=workdir,
      )

    # Log learning rate
    summary_writer.scalar(
        tag='learning_rate',
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(state_list[0].step),
        step=state_list[0].step,
    )

    # SGD step
    state_list, metrics = update_states_jit(
        state_list=state_list,
        batch=train_ds,
        prng_key=next(prng_seq),
    )
    # The computed training loss corresponds to the model before update
    summary_writer.scalar(
        tag='train_loss',
        value=metrics['train_loss'],
        step=state_list[0].step - 1,
    )

    if state_list[0].step == 1:
      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

    # Metrics for evaluation
    if state_list[0].step % config.eval_steps == 0:

      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

      elbo_validation_dict = elbo_validation_jit(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
      )
      for k, v in elbo_validation_dict.items():
        summary_writer.scalar(
            tag=f'elbo_{k}',
            value=v.mean(),
            step=state_list[0].step,
        )

    if state_list[0].step % config.checkpoint_steps == 0:
      for state_i, state_name_i in zip(state_list, state_name_list):
        save_checkpoint(
            state=state_i,
            checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
            keep=config.checkpoints_keep,
        )

    # Wait until computations are done before the next step
    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  logging.info('Final training step: %i', state_list[0].step)

  ### Find best eta ###

  # logging.info('Finding best eta...')

  # # Reset random key sequence
  # prng_seq = hk.PRNGSequence(config.seed)

  # # Initialize search with Bayes
  # eta_star = jnp.ones((config.num_groups,))

  # # key_search = next(prng_seq)

  # # Greedy search to find a cut for initialization
  # eta_neg_elpd_dict = {}

  # eta_star_neg_elpd = loss_neg_elpd(
  #     eta=eta_star,
  #     batch=train_ds,
  #     prng_key=next(prng_seq),
  #     # prng_key=key_search,
  #     state_list_vmp=state_list,
  # )
  # eta_neg_elpd_dict['step'] = [0]
  # eta_neg_elpd_dict['neg_elpd'] = [jnp.expand_dims(eta_star_neg_elpd, 0)]
  # step_greedy = 0
  # summary_writer.scalar(
  #     tag='rnd_eff_eta_star_neg_elpd_greedy_alg',
  #     value=eta_star_neg_elpd,
  #     step=step_greedy,
  # )
  # # grid_search = True

  # # while grid_search and step_greedy < config.num_groups:
  # for step_greedy in range(1, 3):
  #   logging.info(step_greedy)
  #   eta_new = np.array(jnp.tile(eta_star, (config.num_groups, 1)))
  #   np.fill_diagonal(eta_new, 1e-4)
  #   eta_new = np.unique(eta_new, axis=0)
  #   eta_new = jnp.array(eta_new)

  #   eta_new_neg_elpd = jax.vmap(lambda eta: loss_neg_elpd(
  #       eta=eta,
  #       batch=train_ds,
  #       prng_key=next(prng_seq),
  #       # prng_key=key_search,
  #       state_list_vmp=state_list,
  #   ))(
  #       eta_new)
  #   eta_neg_elpd_dict['step'] = eta_neg_elpd_dict['step'] + [step_greedy] * len(
  #       eta_new_neg_elpd)
  #   eta_neg_elpd_dict['neg_elpd'].append(eta_new_neg_elpd)
  #   if (eta_new_neg_elpd.min() < eta_star_neg_elpd) and (
  #       eta_new_neg_elpd.min() <
  #       eta_new_neg_elpd.mean() - eta_new_neg_elpd.std()):
  #     eta_star_neg_elpd = eta_new_neg_elpd.min()
  #     eta_star = eta_new[eta_new_neg_elpd.argmin()]
  #     # step_greedy = step_greedy + 1
  #     summary_writer.scalar(
  #         tag='rnd_eff_eta_star_neg_elpd_greedy_alg',
  #         value=eta_star_neg_elpd,
  #         step=step_greedy,
  #     )
  #   # else:
  #   #   grid_search = False

  # eta_neg_elpd_dict['step'] = np.array(eta_neg_elpd_dict['step'])
  # eta_neg_elpd_dict['neg_elpd'] = np.concatenate(eta_neg_elpd_dict['neg_elpd'])

  # # best = [
  # #     min(eta_neg_elpd_dict['neg_elpd'][eta_neg_elpd_dict['step'] == step_i])
  # #     for step_i in range(eta_neg_elpd_dict['step'].max())
  # # ]
  # # best = best[:(np.where(np.diff(best) > 0)[0][0])]

  # # plot_name = 'greedy_search_eta_star_neg_elpd'
  # # fig, axs = plt.subplots(1, 1, figsize=(10, 5))
  # # axs.scatter(
  # #     x=eta_neg_elpd_dict['step'] + 0.2 *
  # #     jax.random.uniform(next(prng_seq),
  # #                        (len(eta_neg_elpd_dict['step']),)) - 0.1,
  # #     y=eta_neg_elpd_dict['neg_elpd'],
  # #     alpha=0.1,
  # # )
  # # axs.scatter(x=jnp.arange(3), y=jnp.stack(best), color='red')
  # # axs.plot(jnp.arange(3), jnp.stack(best), color='red')
  # # axs.set_xlabel('step')
  # # axs.set_ylabel('neg_elpd')
  # # axs.set_title('Greedy search for best eta')
  # # fig.savefig(pathlib.Path(workdir) / (plot_name + ".png"))

  # # eta_star = jnp.array(config.eta_plot)[2, :]
  # # eta_star = jax.random.beta(
  # #     key=next(prng_seq),
  # #     a=1.,
  # #     b=1.,
  # #     shape=(config.num_groups,),
  # # )

  # # SGD over elpd #
  # eta_star_state = TrainState(
  #     params=eta_star,
  #     opt_state=make_optimizer_eta(**config.optim_kwargs_eta).init(eta_star),
  #     step=0,
  # )
  # for _ in range(config.eta_star_steps):
  #   eta_star_state, neg_elpd = update_eta_star_state(
  #       eta_star_state,
  #       batch=train_ds,
  #       prng_key=next(prng_seq),
  #   )
  #   # Clip eta_star to [0,1] hypercube
  #   eta_star_state = TrainState(
  #       params=jnp.clip(eta_star_state.params, 0., 1.),
  #       opt_state=eta_star_state.opt_state,
  #       step=eta_star_state.step,
  #   )
  #   summary_writer.scalar(
  #       tag='rnd_eff_eta_star_neg_elpd',
  #       value=neg_elpd['train_loss'],
  #       step=eta_star_state.step - 1,
  #   )
  #   for i, eta_star_i in enumerate(eta_star_state.params):
  #     summary_writer.scalar(
  #         tag=f'rnd_eff_eta_star_{i}',
  #         value=eta_star_i,
  #         step=eta_star_state.step - 1,
  #     )

  # Saving checkpoint at the end of the training process
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
      prng_key=next(prng_seq),
      config=config,
      show_elpd=True,
      eta_grid_len=10,
      y_new=y_new,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )

  # maximize_elpd(
  #     state_list=state_list,
  #     batch=train_ds,
  #     prng_key=next(prng_seq),
  #     config=config,
  #     num_samples_elpd=config.num_samples_elpd,
  # )

  return state_list
