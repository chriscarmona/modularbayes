"""Training a Variational Meta-Posterior, using the VMP map."""

import pathlib

from absl import logging

import numpy as np

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
from log_prob_fun import (logprob_joint, sample_eta_values, ModelParams,
                          ModelParamsCut, SmiEta)
import plot
from train_flow import init_state_tuple, load_data, sample_q_as_az

import modularbayes
from modularbayes import (elbo_smi_vmpmap, flatten_dict, plot_to_image,
                          train_step)
from modularbayes._src.typing import (Any, Array, Callable, ConfigDict, Dict,
                                      Optional, PRNGKey, SmiPosteriorStates,
                                      Tuple)

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


def print_trainable_params(
    state_tuple: Tuple[TrainState],
    config: ConfigDict,
    lambda_init_tuple: Tuple,
) -> None:
  """Print a summary of the trainable parameters."""
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda state_i, lambda_init_i: state_i.apply_fn(
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
  summary = tabulate_fn_(state_tuple[0], lambda_init_tuple[0])
  logging.info('VMP-MAP no-cut parameters:')
  for line in summary.split("\n"):
    logging.info(line)

  summary = tabulate_fn_(state_tuple[1], lambda_init_tuple[1])
  logging.info('VMP-MAP cut parameters:')
  for line in summary.split("\n"):
    logging.info(line)


def loss_fn(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  # Compute ELBO.
  elbo_dict = elbo_smi_vmpmap(alpha_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2']))

  return loss_avg


def log_images(
    state_tuple: Tuple[TrainState],
    prng_key: PRNGKey,
    config: ConfigDict,
    dataset: Dict[str, Any],
    num_samples_plot: int,
    vmpmap_fn: hk.Transformed,
    lambda_init_tuple: Tuple[hk.Params],
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    vmpmap_curves_idx: Tuple[int],
    summary_writer: Optional[tensorboard.SummaryWriter] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  assert len(config.smi_eta_cancer_plot) > 0, 'No eta values to plot'
  assert all((x >= 0.0) and (x <= 1.0)
             for x in config.smi_eta_cancer_plot), 'Invalid eta values'

  # We can obtain the variational parameters for all eta values at once
  smi_etas = SmiEta(
      hpv=jnp.ones(len(config.smi_eta_cancer_plot)),
      cancer=jnp.array(config.smi_eta_cancer_plot),
  )
  eta_values = (
      smi_etas[0] if len(smi_etas) == 1 else jnp.stack(smi_etas, axis=-1))
  # Produce flow parameters as a function of eta
  lambda_tuple = [
      state_i.apply_fn(
          state_i.params,
          eta_values=eta_values,
          lambda_init=lambda_i,
      ) for state_i, lambda_i in zip(state_tuple, lambda_init_tuple)
  ]

  # Sample from flows and plot, one eta at a time
  for i, eta_i in enumerate(config.smi_eta_cancer_plot):
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
        show_phi_trace=False,
        show_theta_trace=False,
        show_loglinear_scatter=True,
        show_theta_pairplot=True,
        eta=eta_i,
        suffix=f"_eta_cancer_{float(eta_i):.3f}",
        step=state_tuple[0].step,
        workdir_png=workdir_png,
        summary_writer=summary_writer,
    )

  # Plot a few curves representing the VMP map for the no-cut parameters
  model_name = 'epidemiology'
  fig, _ = plot.plot_vmp_map(
      alpha=state_tuple[0].params,
      vmpmap_fn=vmpmap_fn,
      lambda_init=lambda_init_tuple[0],
      lambda_idx_to_plot=vmpmap_curves_idx,
      eta_cancer_grid=jnp.linspace(0.0, 1.0, 101).round(3),
  )
  fig.suptitle("Example of VMP map curves")
  fig.tight_layout()
  if workdir_png:
    plot_name = f"{model_name}_vmpmap_example_curves"
    fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
  image = plot_to_image(fig)
  if summary_writer:
    plot_name = f"{model_name}_vmpmap_example_curves"
    summary_writer.image(
        tag=plot_name,
        image=image,
        step=state_tuple[0].step,
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

  # Full dataset used everytime
  # Small data, no need to batch
  train_ds = load_data()

  # phi_dim and theta_dim are also arguments of the flow,
  # as they define its dimension
  config.flow_kwargs.phi_dim = train_ds['Z'].shape[0]
  config.flow_kwargs.theta_dim = 2
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

  # To initialize the VMP-map, we need one example of its output
  # The output of the VMP-map is lambda, the parameters of the variational posterior
  lambda_init_tuple = init_state_tuple(
      prng_key=next(prng_seq),
      config=config,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      sample_shape=(config.num_samples_elbo,),
      eta_values=None,
  )
  lambda_init_tuple = [x.params for x in lambda_init_tuple]

  # Define function that produce a tuple of lambda (flow parameters)
  @hk.without_apply_rng
  @hk.transform
  def vmpmap_fn(eta_values, lambda_init):
    vmpmap = getattr(modularbayes, config.vmp_map_name)(
        **config.vmp_map_kwargs, params_flow_init=lambda_init)
    lambda_out = vmpmap(eta_values)
    return lambda_out

  ### Initialise Variational Meta-Posterior Map ###
  params_tuple_ = [
      vmpmap_fn.init(
          next(prng_seq),
          eta_values=jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
          lambda_init=lambda_init_) for lambda_init_ in lambda_init_tuple
  ]
  state_tuple = SmiPosteriorStates(*[
      TrainState.create(
          apply_fn=vmpmap_fn.apply,
          params=params_,
          tx=make_optimizer(**config.optim_kwargs),
      ) for params_ in params_tuple_
  ])

  print_trainable_params(
      state_tuple=state_tuple,
      config=config,
      lambda_init_tuple=lambda_init_tuple,
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
            'vmpmap_fn': vmpmap_fn,
            'lambda_init_tuple': lambda_init_tuple,
            'sample_eta_fn': sample_eta_values,
            'sample_eta_kwargs': {
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

  if state_tuple[0].step < config.training_steps:
    logging.info('Training Variational Meta-Posterior (VMP-map)...')

  # Reset random key sequence
  prng_seq = hk.PRNGSequence(config.seed)

  while state_tuple[0].step < config.training_steps:

    # Plots to monitor training
    if (config.log_img_steps
        is not None) and (state_tuple[0].step % config.log_img_steps == 0):
      logging.info("\t\t Logging plots...")
      log_images(
          state_tuple=state_tuple,
          prng_key=next(prng_seq),
          config=config,
          dataset=train_ds,
          num_samples_plot=config.num_samples_plot,
          vmpmap_fn=vmpmap_fn,
          lambda_init_tuple=lambda_init_tuple,
          flow_get_fn_nocut=flow_get_fn_nocut,
          flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
          vmpmap_curves_idx=config.vmpmap_curves_idx,
          summary_writer=summary_writer,
          workdir_png=workdir,
      )
      plt.close()
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

    # The computed training loss would correspond to the model before update
    summary_writer.scalar(
        tag='train_loss',
        value=metrics['train_loss'],
        step=state_tuple[0].step - 1,
    )

    if state_tuple[0].step == 2:
      logging.info("STEP: %5d; training loss: %.3f", state_tuple[0].step - 1,
                   metrics["train_loss"])

    if state_tuple[0].step % config.eval_steps == 0:
      logging.info("STEP: %5d; training loss: %.3f", state_tuple[0].step - 1,
                   metrics["train_loss"])

    # Save checkpoints
    for state, mngr in zip(state_tuple, orbax_ckpt_mngrs):
      mngr.save(step=int(state.step), items=state)

    # Wait until computations are done before the next step
    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  logging.info('Final training step: %i', state_tuple[0].step)

  # Last plot of posteriors
  log_images(
      state_tuple=state_tuple,
      prng_key=next(prng_seq),
      config=config,
      dataset=train_ds,
      num_samples_plot=config.num_samples_plot,
      vmpmap_fn=vmpmap_fn,
      lambda_init_tuple=lambda_init_tuple,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      vmpmap_curves_idx=config.vmpmap_curves_idx,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )
  plt.close()

  return state_tuple


# # For debugging
# config = get_config()
# import pathlib
# workdir = str(pathlib.Path.home() / f'modularbayes-output-exp/epidemiology/nsf/vmp_map')
# # train_and_evaluate(config, workdir)
