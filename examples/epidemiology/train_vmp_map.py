"""Flow model trained on Epidemiology data."""
import pathlib

from absl import logging

import numpy as np
import scipy

from matplotlib import pyplot as plt

# from clu import metric_writers
from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax

from tensorflow_probability.substrates import jax as tfp

from train_flow import load_dataset, q_distr, elbo_fn

import plot

from modularbayes import metaposterior
from modularbayes import utils
from modularbayes.utils.training import TrainState
from modularbayes.typing import (Any, Array, Batch, ConfigDict, Dict, List,
                                 Mapping, Optional, PRNGKey, SummaryWriter,
                                 Union)

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
  return getattr(metaposterior, vmp_map_name)(
      **vmp_map_kwargs, params_flow_init=params_flow_init)(
          eta)


def loss_pretrain(
    params: hk.Params,
    batch: Batch,  # pylint: disable=unused-argument
    prng_key: PRNGKey,
    num_samples_eta: int,
    vmp_map_name: str,
    vmp_map_kwargs: Dict[str, Any],
    params_flow_init: hk.Params,
    eta_sampling_a: float,
    eta_sampling_b: float,
    eps: float,
):
  """Loss to pretrain VMP map.

  Produce constant variational parameters equal to params_flow_init.
  """

  prng_seq = hk.PRNGSequence(prng_key)

  # Sample eta values
  # eta = jax.random.uniform(key=next(prng_seq), shape=(num_samples_eta, 1))
  eta = jax.random.beta(
      key=next(prng_seq),
      a=eta_sampling_a,
      b=eta_sampling_b,
      shape=(num_samples_eta, 1),
  )

  params_flow_vmap = vmp_map.apply(
      params,
      eta=eta,
      vmp_map_name=vmp_map_name,
      vmp_map_kwargs=vmp_map_kwargs,
      params_flow_init=params_flow_init,
  )

  # Square difference between params_flow_vmap and the target params_flow_init

  # Add noise to the target to break simetries
  noise_tree = jax.tree_map(
      lambda y: eps * jax.random.normal(next(prng_seq), y.shape),
      tree=params_flow_vmap,
  )

  diff_tree = jax.tree_multimap(
      lambda x, y, noise: jnp.square(y - (jnp.broadcast_to(x, y.shape) + noise))
      .sum(),
      params_flow_init,
      params_flow_vmap,
      noise_tree,
  )

  diff_total = jnp.stack(jax.tree_util.tree_leaves(diff_tree)).sum()

  return diff_total


def loss(
    params: hk.Params,
    batch: Batch,
    prng_key: PRNGKey,
    num_samples_eta: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    params_flow_init: hk.Params,
    num_samples_flow: int,
    vmp_map_name: str,
    vmp_map_kwargs: Dict[str, Any],
    eta_sampling_a: float,
    eta_sampling_b: float,
) -> Array:
  """Define training loss function."""

  # Sample eta values
  prng_key, key_eta = jax.random.split(prng_key)

  eta = jax.random.beta(
      key=key_eta,
      a=eta_sampling_a,
      b=eta_sampling_b,
      shape=(num_samples_eta, 1),
  )

  smi_eta_vmap = {
      'modules': jnp.concatenate([jnp.ones((num_samples_eta, 1)), eta], axis=-1)
  }

  params_flow_vmap = vmp_map.apply(
      params,
      eta=eta,
      vmp_map_name=vmp_map_name,
      vmp_map_kwargs=vmp_map_kwargs,
      params_flow_init=params_flow_init,
  )
  # globals().update(vmp_map_kwargs)

  # Use same key for every eta
  # (same key implies same samples from the base distribution)
  prng_key, key_flow = jax.random.split(prng_key)

  # Sample from posterior
  q_distr_out_vmap = jax.vmap(lambda params_flow: hk.transform(q_distr).apply(
      params_flow,
      key_flow,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples_flow,)))(
          params_flow_vmap)

  # Compute ELBO.
  elbo_dict = jax.vmap(lambda q_distr_out_vmap_i, smi_eta_i: elbo_fn(
      q_distr_out=q_distr_out_vmap_i,
      smi_eta=smi_eta_i,
      batch=batch,
  ))(q_distr_out_vmap, smi_eta_vmap)

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

  # When only two lambdas are plotted, we include the relationship between them
  # in the first axis
  idx_offset = 1 if isinstance(lambda_idx, List) & (len(lambda_idx) == 2) else 0
  fig, axs = plt.subplots(
      nrows=1,
      ncols=len(lambda_idx) + idx_offset,
      figsize=(4 * (len(lambda_idx) + idx_offset), 3),
  )
  if not lambda_all.shape[1] > 0:
    return fig, axs

  if idx_offset == 1:
    i, j = lambda_idx
    axs[0].plot(lambda_all[:, i], lambda_all[:, j])
    axs[0].set_xlabel(f'lambda_{i}')
    axs[0].set_ylabel(f'lambda_{j}')

  # Plot vmp-map as a function of eta
  for i, idx_i in enumerate(lambda_idx):
    axs[i + idx_offset].plot(eta_grid, lambda_all[:, idx_i])
    axs[i + idx_offset].set_xlabel('eta')
    axs[i + idx_offset].set_ylabel(f'lambda_{idx_i}')

  fig.tight_layout()

  return fig, axs


def log_images(
    state: TrainState,
    prng_key: PRNGKey,
    config: ConfigDict,
    eta_grid_len: int,
    params_flow_init: hk.Params,
    summary_writer: Optional[SummaryWriter],
    workdir_png: Optional[str],
) -> None:
  """Plots to monitor during training."""

  eta_plot = jnp.array(config.eta_plot).reshape(-1, 1)

  assert eta_plot.ndim == 2
  assert eta_plot.shape[-1] == 1

  # Produce flow parameters as a function of eta
  params_flow_vmap = vmp_map.apply(
      state.params,
      eta=eta_plot,
      vmp_map_name=config.vmp_map_name,
      vmp_map_kwargs=config.vmp_map_kwargs,
      params_flow_init=params_flow_init,
  )

  # Use same key for every eta
  # (same key implies same samples from the base distribution)
  prng_key, key_flow = jax.random.split(prng_key)

  # Sample from posterior
  q_distr_out = jax.vmap(lambda params_flow: hk.transform(q_distr).apply(
      params_flow,
      key_flow,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_plot,)))(
          params_flow_vmap)

  # Plot relation: prevalence (lambda) vs incidence (phi)
  for i, eta_y_i in enumerate(eta_plot.squeeze()):
    posterior_sample_dict_i = jax.tree_util.tree_map(
        lambda x: x[i],  # pylint: disable=cell-var-from-loop
        q_distr_out['posterior_sample'])
    plot.posterior_samples(
        posterior_sample_dict=posterior_sample_dict_i,
        summary_writer=summary_writer,
        step=state.step,
        eta=eta_y_i,
        workdir_png=workdir_png,
    )

  # Visualize meta-posterior map
  eta_grid = jnp.linspace(0, 1, eta_grid_len).reshape(-1, 1)

  if config.flow_name == 'mean_field':
    # Plot locations of theta_1 and theta_2
    plot_name = 'epidemiology_vmp_map_theta'
    fig, axs = plot_vmp_map(
        state=state,
        vmp_map_name=config.vmp_map_name,
        vmp_map_kwargs=config.vmp_map_kwargs,
        params_flow_init=params_flow_init,
        lambda_idx=[26, 27],  # lambda_26 = theta_1_loc, lambda_27 = theta_2_loc
        eta_grid=eta_grid,
        constant_lambda_ignore_plot=False,
    )
    axs[0].set_xlabel('theta_1_loc')
    axs[0].set_ylabel('theta_2_loc')
    axs[1].set_ylabel('theta_1_loc')
    axs[2].set_ylabel('theta_2_loc')
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    if summary_writer:
      summary_writer.image(plot_name, utils.plot_to_image(fig), step=state.step)

  plot_name = 'epidemiology_vmp_map'
  fig, axs = plot_vmp_map(
      state=state,
      vmp_map_name=config.vmp_map_name,
      vmp_map_kwargs=config.vmp_map_kwargs,
      params_flow_init=params_flow_init,
      lambda_idx=config.lambda_idx_plot,
      eta_grid=eta_grid,
      constant_lambda_ignore_plot=config.constant_lambda_ignore_plot,
  )
  if workdir_png:
    fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
  if summary_writer:
    summary_writer.image(plot_name, utils.plot_to_image(fig), step=state.step)


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

  # smi_eta = {'modules':jnp.ones((1,2))}
  # elbo_fn(config.params_flow, next(prng_seq), train_ds, smi_eta)

  # Get an example of the output tree to be produced by the meta function
  if config.state_flow_init_path == '':
    params_flow_init = hk.transform(q_distr).init(
        next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        sample_shape=(config.num_samples_elbo,),
    )
  else:
    state_flow_init = utils.load_ckpt(path=config.state_flow_init_path)
    params_flow_init = state_flow_init.params

  ### Set Variational Meta-Posterior Map ###

  # eta knots
  if config.vmp_map_name in ['VmpGP', 'VmpCubicSpline']:
    # grid of percentile values from a beta distribution
    config.vmp_map_kwargs['eta_knots'] = scipy.stats.beta.ppf(
        jnp.linspace(0, 1., config.vmp_map_kwargs.num_knots).reshape(-1, 1),
        a=config.eta_sampling_a,
        b=config.eta_sampling_b,
    ).tolist()
    del config.vmp_map_kwargs['num_knots']

  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')
  state = utils.initial_state_ckpt(
      checkpoint_dir=checkpoint_dir,
      forward_fn=vmp_map,
      forward_fn_kwargs={
          'eta': jnp.ones((config.num_samples_eta, 1)),
          'vmp_map_name': config.vmp_map_name,
          'vmp_map_kwargs': config.vmp_map_kwargs,
          'params_flow_init': params_flow_init,
      },
      prng_key=next(prng_seq),
      optimizer=make_optimizer(**config.optim_kwargs),
  )
  # globals().update(forward_fn_kwargs)

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)
  if jax.process_index() == 0 and state.step < config.training_steps:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(utils.flatten_dict(config))
  else:
    summary_writer = None

  # Print a useful summary of the execution of the VHP-map architecture.
  summary = hk.experimental.tabulate(
      f=lambda state_i, params_flow_init_i: vmp_map.apply(
          state_i.params,
          eta=jnp.ones((config.num_samples_eta, 1)),
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
  )(state, params_flow_init)
  for line in summary.split("\n"):
    logging.info(line)

  ### Pretaining VMP map ###
  if state.step == 0:
    logging.info('Pre-training VMP-map...')
    update_state_jit = lambda state, prng_key: utils.update_state(
        state=state,
        batch=None,
        prng_key=prng_key,
        optimizer=make_optimizer(**config.optim_kwargs),
        loss_fn=loss_pretrain,
        loss_fn_kwargs={
            'num_samples_eta': config.num_samples_eta,
            'vmp_map_name': config.vmp_map_name,
            'vmp_map_kwargs': config.vmp_map_kwargs,
            'params_flow_init': params_flow_init,
            'eta_sampling_a': config.eta_sampling_a,
            'eta_sampling_b': config.eta_sampling_b,
            'eps': config.eps_noise_pretrain,
        },
    )
    # globals().update(loss_fn_kwargs)
    update_state_jit = jax.jit(update_state_jit)

    metrics = {'pretrain_loss': jnp.inf}
    while metrics['pretrain_loss'] > config.pretrain_error:
      state, metrics = update_state_jit(
          state=state,
          prng_key=next(prng_seq),
      )
      metrics['pretrain_loss'] = metrics['train_loss']
      del metrics['train_loss']

      summary_writer.scalar('pretrain_loss', metrics['pretrain_loss'],
                            state.step - 1)

      if ((state.step - 1) % config.eval_steps
          == 0) or (metrics['pretrain_loss'] <= config.pretrain_error):
        logging.info("STEP: %5d; pretraining loss: %.3f", state.step - 1,
                     metrics["pretrain_loss"])

    # Reset other state attributes
    state = TrainState(
        params=state.params,
        opt_state=make_optimizer(**config.optim_kwargs).init(state.params),
        step=1,
    )
    # Save pretrained model
    utils.save_checkpoint(
        state=state,
        checkpoint_dir=checkpoint_dir,
        keep=config.checkpoints_keep,
    )
    logging.info('Pre-training completed!')

  ### Training VMP map ###
  update_state_jit = lambda state, batch, prng_key: utils.update_state(
      state=state,
      batch=batch,
      prng_key=prng_key,
      optimizer=make_optimizer(**config.optim_kwargs),
      loss_fn=loss,
      loss_fn_kwargs={
          'num_samples_eta': config.num_samples_eta,
          'flow_name': config.flow_name,
          'flow_kwargs': config.flow_kwargs,
          'num_samples_flow': config.num_samples_elbo,
          'vmp_map_name': config.vmp_map_name,
          'vmp_map_kwargs': config.vmp_map_kwargs,
          'params_flow_init': params_flow_init,
          'eta_sampling_a': config.eta_sampling_a,
          'eta_sampling_b': config.eta_sampling_b,
      },
  )
  # globals().update(loss_fn_kwargs)
  update_state_jit = jax.jit(update_state_jit)

  if state.step < config.training_steps:
    logging.info('Training VMP-map...')

  while state.step < config.training_steps:

    # Plots to monitor training
    if (config.log_img_steps
        is not None) and ((state.step in [0, 1]) or
                          (state.step % config.log_img_steps == 0)):
      # print("Logging images...\n")
      log_images(
          state=state,
          prng_key=next(prng_seq),
          config=config,
          eta_grid_len=51,
          params_flow_init=params_flow_init,
          summary_writer=summary_writer,
          workdir_png=workdir,
      )
      plt.close()

    # Log learning rate
    summary_writer.scalar(
        tag='learning_rate',
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(state.step),
        step=state.step,
    )

    state, metrics = update_state_jit(
        state=state,
        batch=train_ds,
        prng_key=next(prng_seq),
    )
    # The computed training loss would correspond to the model before update
    summary_writer.scalar(
        tag='train_loss',
        value=metrics['train_loss'],
        step=state.step - 1,
    )

    if state.step == 2:
      logging.info("STEP: %5d; training loss: %.3f", state.step - 1,
                   metrics["train_loss"])

    if state.step % config.eval_steps == 0:
      logging.info("STEP: %5d; training loss: %.3f", state.step - 1,
                   metrics["train_loss"])

    if (state.step) % config.checkpoint_steps == 0:
      utils.save_checkpoint(
          state=state,
          checkpoint_dir=checkpoint_dir,
          keep=config.checkpoints_keep,
      )

    # Wait until computations are done before the next step
    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  logging.info('Final training step: %i', state.step)

  # Saving checkpoint at the end of the training process
  # (in case training_steps is not multiple of checkpoint_steps)
  utils.save_checkpoint(
      state=state,
      checkpoint_dir=checkpoint_dir,
      keep=config.checkpoints_keep,
  )

  # Last plot of posteriors
  log_images(
      state=state,
      prng_key=jax.random.PRNGKey(config.seed),
      config=config,
      eta_grid_len=51,
      params_flow_init=params_flow_init,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )
  plt.close()

  return state


# # For debugging
# config = get_config()
# workdir = pathlib.Path.home() / 'smi/output/debug'
# config.state_flow_init_path = str(pathlib.Path.home() /
#                                   ('smi/output/epidemiology/mean_field/' +
#                                    'eta_0.100/checkpoints/ckpt_010000'))
# train_and_evaluate(config, workdir)
