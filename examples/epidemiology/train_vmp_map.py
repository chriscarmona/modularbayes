"""Flow model trained on Epidemiology data."""
import pathlib

from absl import logging

import numpy as np

from matplotlib import pyplot as plt

from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax

from tensorflow_probability.substrates import jax as tfp

import plot
from train_flow import (load_dataset, sample_all_flows, q_distr_phi,
                        q_distr_theta, elbo_estimate)

import modularbayes
from modularbayes._src.utils.training import TrainState
from modularbayes import (flatten_dict, plot_to_image, initial_state_ckpt,
                          update_states, save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict, List,
                                      Mapping, Optional, PRNGKey, SummaryWriter,
                                      Tuple, Union)

kernels = tfp.math.psd_kernels

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

  prng_seq = hk.PRNGSequence(prng_key)

  # Sample eta values (only for Y module)
  key_eta = next(prng_seq)
  etas = jax.random.beta(
      key=key_eta,
      a=eta_sampling_a,
      b=eta_sampling_b,
      shape=(num_samples_eta, eta_dim),
  )
  # Set eta_z=1
  etas = jnp.concatenate([jnp.ones_like(etas), etas], axis=-1)
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
  key_elbo = next(prng_seq)

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
    constant_lambda_ignore_plot: bool,
):
  """Visualize VMP map."""

  assert eta_grid.ndim == 2
  assert eta_grid.shape[-1] == 2
  eta_grid_len, _ = eta_grid.shape

  params_flow_grid = vmp_map.apply(
      state.params,
      eta=eta_grid,
      vmp_map_name=vmp_map_name,
      vmp_map_kwargs=vmp_map_kwargs,
      params_flow_init=params_flow_init,
  )
  # All variational parameters
  lambda_all = jnp.concatenate([
      x.reshape(eta_grid_len, -1)
      for x in jax.tree_util.tree_leaves(params_flow_grid)
  ],
                               axis=-1)
  # Ignore flat functions of eta
  if constant_lambda_ignore_plot:
    lambda_all = lambda_all[:,
                            jnp.where(
                                jnp.square(lambda_all - lambda_all[[0], :]).sum(
                                    axis=0) > 0.)[0]]

  # Plot variational parameters as a function of eta_Y
  fig, axs = plt.subplots(
      nrows=1,
      ncols=len(lambda_idx),
      figsize=(4 * len(lambda_idx), 3),
      squeeze=False,
  )
  if not lambda_all.shape[1] > 0:
    return fig, axs

  for i, idx_i in enumerate(lambda_idx):
    axs[0, i].plot(eta_grid[:, 1], lambda_all[:, idx_i])
    axs[0, i].plot(eta_grid[:, 1], lambda_all[:, idx_i])
    axs[0, i].set_xlabel('eta')
    axs[0, i].set_ylabel(f'lambda_{idx_i}')
  fig.tight_layout()

  return fig, axs


def log_images(
    state_list: List[TrainState],
    prng_key: PRNGKey,
    config: ConfigDict,
    state_name_list: List[str],
    eta_grid_len: int,
    params_flow_init_list: List[hk.Params],
    summary_writer: Optional[SummaryWriter],
    workdir_png: Optional[str],
) -> None:
  """Plots to monitor during training."""

  eta_plot = jnp.array(config.eta_plot)

  assert eta_plot.ndim == 2
  assert eta_plot.shape[-1] == 2

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
  prng_key, key_flow = jax.random.split(prng_key)

  # Sample from flow
  q_distr_out = jax.vmap(lambda params_flow_tuple: sample_all_flows(
      params_tuple=params_flow_tuple,
      prng_key=key_flow,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_plot,),
  ))(
      params_flow_tuple)

  # Plot relation: prevalence (lambda) vs incidence (phi)
  for i, eta_i in enumerate(eta_plot):
    posterior_sample_dict_i = jax.tree_util.tree_map(
        lambda x: x[i],  # pylint: disable=cell-var-from-loop
        q_distr_out['posterior_sample'])
    plot.posterior_samples(
        posterior_sample_dict=posterior_sample_dict_i,
        summary_writer=summary_writer,
        step=state_list[0].step,
        eta=float(eta_i[1]),
        workdir_png=workdir_png,
    )

  # Visualize meta-posterior map
  eta_grid = jnp.linspace(0, 1, eta_grid_len).reshape(-1, 1)
  eta_grid = jnp.concatenate([jnp.ones_like(eta_grid), eta_grid], axis=-1)

  for state_i, state_name_i, params_flow_init_i in zip(state_list,
                                                       state_name_list,
                                                       params_flow_init_list):

    # state_i = state_list[1]
    # state_name_i = state_name_list[1]
    # params_flow_init_i = params_flow_init_list[1]

    cond_theta_easy = (config.flow_name == 'mean_field') and (state_name_i in [
        'theta', 'theta_aux'
    ])

    plot_name = 'epidemiology_vmp_map_' + state_name_i
    fig, axs = plot_vmp_map(
        state=state_i,
        vmp_map_name=config.vmp_map_name,
        vmp_map_kwargs=config.vmp_map_kwargs,
        params_flow_init=params_flow_init_i,
        lambda_idx=[0, 1] if cond_theta_easy else config.lambda_idx_plot,
        eta_grid=eta_grid,
        constant_lambda_ignore_plot=config.constant_lambda_ignore_plot,
    )
    if cond_theta_easy:
      axs[0, 0].set_ylabel('theta_1_loc')
      axs[0, 1].set_ylabel('theta_2_loc')
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      summary_writer.image(plot_name, plot_to_image(fig), step=state_i.step)


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
  train_ds = load_dataset()

  # phi_dim and theta_dim are also arguments of the flow,
  # as they define its dimension
  config.flow_kwargs.phi_dim = train_ds['Z'].shape[0]
  config.flow_kwargs.theta_dim = 2
  config.flow_kwargs.is_smi = True

  state_name_list = ['phi', 'theta', 'theta_aux']

  # smi_eta = {'modules':jnp.ones((1,2))}
  # elbo_fn(config.params_flow, next(prng_seq), train_ds, smi_eta)

  # Get an example of the output tree to be produced by the meta function
  params_flow_init_list = []

  params_flow_init_list.append(
      hk.transform(q_distr_phi).init(
          next(prng_seq),
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          sample_shape=(config.num_samples_elbo,),
      ))

  # Get an initial sample of phi
  # (used below to initialize beta and tau)
  phi_base_sample_init = hk.transform(q_distr_phi).apply(
      params_flow_init_list[0],
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_elbo,),
  )['phi_base_sample']

  params_flow_init_list.append(
      hk.transform(q_distr_theta).init(
          next(prng_seq),
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          phi_base_sample=phi_base_sample_init,
          is_aux=False,
      ))

  params_flow_init_list.append(
      hk.transform(q_distr_theta).init(
          next(prng_seq),
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          phi_base_sample=phi_base_sample_init,
          is_aux=True,
      ))

  ### Set Variational Meta-Posterior Map ###
  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')

  state_list = [
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_i}',
          forward_fn=vmp_map,
          forward_fn_kwargs={
              'eta': jnp.ones((config.num_samples_eta, 2)),
              'vmp_map_name': config.vmp_map_name,
              'vmp_map_kwargs': config.vmp_map_kwargs,
              'params_flow_init': params_flow_init_i,
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ) for state_name_i, params_flow_init_i in zip(state_name_list,
                                                    params_flow_init_list)
  ]

  # globals().update(forward_fn_kwargs)

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
          eta=jnp.ones((config.num_samples_eta, 2)),
          vmp_map_name=config.vmp_map_name,
          vmp_map_kwargs=config.vmp_map_kwargs,
          params_flow_init=params_flow_init_i),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_list[0], params_flow_init_list[0])
  logging.info('VMP-MAP PHI:')
  for line in summary.split("\n"):
    logging.info(line)

  summary = tabulate_fn_(state_list[1], params_flow_init_list[1])
  logging.info('VMP-MAP THETA:')
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
          'eta_name': 'modules',
          'eta_dim': 1,
          'eta_sampling_a': config.eta_sampling_a,
          'eta_sampling_b': config.eta_sampling_b,
      },
  )
  # globals().update(loss_fn_kwargs)
  update_states_jit = jax.jit(update_states_jit)

  if state_list[0].step < config.training_steps:
    logging.info('Training Variational Meta-Posterior (VMP-map)...')

  # Reset random key sequence
  prng_seq = hk.PRNGSequence(config.seed)

  while state_list[0].step < config.training_steps:

    # Plots to monitor training
    if (config.log_img_steps
        is not None) and ((state_list[0].step == 1) or
                          (state_list[0].step % config.log_img_steps == 0)):
      # print("Logging images...\n")
      log_images(
          state_list=state_list,
          prng_key=next(prng_seq),
          config=config,
          state_name_list=state_name_list,
          eta_grid_len=51,
          params_flow_init_list=params_flow_init_list,
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
      prng_key=jax.random.PRNGKey(config.seed),
      config=config,
      state_name_list=state_name_list,
      eta_grid_len=51,
      params_flow_init_list=params_flow_init_list,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )
  plt.close()

  return state_list
