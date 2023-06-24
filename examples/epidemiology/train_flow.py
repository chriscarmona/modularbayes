"""Training a Normalizing Flow."""

import pathlib

from absl import logging

import numpy as np
from matplotlib import pyplot as plt

from arviz import InferenceData

import jax
from jax import numpy as jnp
from flax.metrics import tensorboard
from flax.training.train_state import TrainState
import haiku as hk
import optax
import orbax.checkpoint

import data
import flows
from flows import split_flow_nocut, split_flow_cut
from log_prob_fun import logprob_joint, ModelParams, ModelParamsCut, SmiEta
import plot

from modularbayes import (elbo_smi, flatten_dict, sample_q_nocut,
                          sample_q_cutgivennocut, sample_q, train_step)
from modularbayes._src.typing import (Any, Array, Callable, ConfigDict, Dict,
                                      Optional, PRNGKey, SmiPosteriorStates,
                                      Tuple)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def load_data() -> Dict[str, Array]:
  """Load Epidemiology data from Plummer."""
  dataset_dict = dict(
      zip(['Z', 'N', 'Y', 'T'],
          np.split(data.epidemiology.to_numpy(), 4, axis=-1)))
  dataset_dict = {
      key: jnp.array(value.squeeze()) for key, value in dataset_dict.items()
  }
  return dataset_dict


def init_state_tuple(
    prng_key: PRNGKey,
    config: ConfigDict,
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    sample_shape: Tuple[int],
    eta_values: Array,
):
  """Initialize states of the three normalizing flows."""
  prng_seq = hk.PRNGSequence(prng_key)

  state_list = []

  # Initialize state of nocut model parameters
  lambda_nocut_ = sample_q_nocut.init(
      next(prng_seq),
      flow_get_fn=flow_get_fn_nocut,
      flow_kwargs=config.flow_kwargs,
      split_flow_fn=split_flow_nocut,
      sample_shape=sample_shape,
      eta_values=eta_values,
  )
  state_list.append(
      TrainState.create(
          apply_fn=sample_q_nocut.apply,
          params=lambda_nocut_,
          tx=make_optimizer(**config.optim_kwargs),
      ))

  # Get an initial sample of the base distribution of nocut model params
  nocut_base_sample_ = state_list[0].apply_fn(
      state_list[0].params,
      next(prng_seq),
      flow_get_fn=flow_get_fn_nocut,
      flow_kwargs=config.flow_kwargs,
      split_flow_fn=split_flow_nocut,
      sample_shape=sample_shape,
      eta_values=eta_values,
  )['sample_base']

  # Initialize state of cut model parameters
  lambda_cut_ = sample_q_cutgivennocut.init(
      next(prng_seq),
      flow_get_fn=flow_get_fn_cutgivennocut,
      flow_kwargs=config.flow_kwargs,
      split_flow_fn=split_flow_cut,
      nocut_base_sample=nocut_base_sample_,
      eta_values=eta_values,
  )
  state_list.append(
      TrainState.create(
          apply_fn=sample_q_cutgivennocut.apply,
          params=lambda_cut_,
          tx=make_optimizer(**config.optim_kwargs),
      ))

  # Initialize state of cut model parameters, the auxiliary copy
  state_list.append(
      TrainState.create(
          apply_fn=sample_q_cutgivennocut.apply,
          params=lambda_cut_,
          tx=make_optimizer(**config.optim_kwargs),
      ))

  # Simply transform the list into a named tuple
  state_tuple = SmiPosteriorStates(*state_list)

  return state_tuple


def print_trainable_params(
    state_tuple: Tuple[TrainState],
    prng_key: PRNGKey,
    config: ConfigDict,
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    sample_shape: Tuple[int],
    eta_values: Array,
) -> None:
  """Print a summary of the trainable parameters."""
  prng_seq = hk.PRNGSequence(prng_key)

  logging.info('\nFlow no-cut parameters:\n')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda state_, prng_key_: state_.apply_fn(
          state_.params,
          prng_key_,
          flow_get_fn=flow_get_fn_nocut,
          flow_kwargs=config.flow_kwargs,
          split_flow_fn=split_flow_nocut,
          sample_shape=sample_shape,
          eta_values=eta_values,
      ),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_tuple[0], next(prng_seq))
  for line in summary.split("\n"):
    logging.info(line)

  # Get an initial sample of the base distribution of nocut params
  nocut_base_sample_ = state_tuple[0].apply_fn(
      state_tuple[0].params,
      next(prng_seq),
      flow_get_fn=flow_get_fn_nocut,
      flow_kwargs=config.flow_kwargs,
      split_flow_fn=split_flow_nocut,
      sample_shape=sample_shape,
      eta_values=eta_values,
  )['sample_base']

  logging.info('\nFlow cut parameters:\n')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda state_, prng_key_: state_.apply_fn(
          state_.params,
          prng_key_,
          flow_get_fn=flow_get_fn_cutgivennocut,
          flow_kwargs=config.flow_kwargs,
          split_flow_fn=split_flow_cut,
          nocut_base_sample=nocut_base_sample_,
          eta_values=eta_values,
      ),
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )
  summary = tabulate_fn_(state_tuple[1], next(prng_seq))
  for line in summary.split("\n"):
    logging.info(line)


def make_optimizer(
    lr_schedule_name,
    lr_schedule_kwargs,
    grad_clip_value,
) -> optax.GradientTransformation:
  """Define optimizer to train the Flow."""
  schedule = getattr(optax, lr_schedule_name)(**lr_schedule_kwargs)

  optimizer = optax.chain(*[
      optax.clip_by_global_norm(max_norm=grad_clip_value),
      optax.adabelief(learning_rate=schedule),
  ])
  return optimizer


def loss_fn(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  ### Compute ELBO ###
  elbo_dict = elbo_smi(lambda_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2']))

  return loss_avg


def sample_q_as_az(
    lambda_tuple: Tuple[hk.Params],
    dataset: Dict[str, Any],
    prng_key: PRNGKey,
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    flow_kwargs: Dict[str, Any],
    sample_shape: Optional[Tuple[int]],
    eta_values: Optional[Array] = None,
) -> InferenceData:
  """Plots to monitor during training."""
  if eta_values is not None:
    assert eta_values.ndim == 2
    assert (eta_values.shape[0],) == sample_shape
  # Sample from flow
  q_distr_out = sample_q(
      lambda_tuple=lambda_tuple,
      prng_key=prng_key,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      flow_kwargs=flow_kwargs,
      model_params_tupleclass=ModelParams,
      split_flow_fn_nocut=split_flow_nocut,
      split_flow_fn_cut=split_flow_cut,
      sample_shape=sample_shape,
      eta_values=eta_values,
  )
  # Add dimension for "chains"
  model_params_az = jax.tree_map(lambda x: x[None, ...],
                                 q_distr_out['model_params_sample'])
  # Create InferenceData object
  az_data = plot.arviz_from_samples(
      dataset=dataset,
      model_params=model_params_az,
  )
  return az_data


def train_and_evaluate(config: ConfigDict, workdir: str) -> TrainState:
  """Model training and evaluation.

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

  # In general, it would be possible to modulate the influence of both modules
  # for now, we only focus on the influence of the cancer module
  smi_eta = SmiEta(hpv=1.0, cancer=config.smi_eta_cancer)

  phi_dim = train_ds['Z'].shape[0]
  theta_dim = 2

  # phi_dim and theta_dim are also arguments of the flow,
  # as they define its dimension
  config.flow_kwargs.phi_dim = phi_dim
  config.flow_kwargs.theta_dim = theta_dim
  # Also is_smi modifies the dimension of the flow, due to the duplicated params
  config.flow_kwargs.is_smi = (smi_eta is not None)

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
      eta_values=None,
  )

  # Print a summary of the flow architecture
  print_trainable_params(
      state_tuple=state_tuple,
      prng_key=next(prng_seq),
      config=config,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      sample_shape=(config.num_samples_elbo,),
      eta_values=None,
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
            'smi_eta': smi_eta,
        },
    )

  @jax.jit
  def elbo_validation_jit(state_tuple, batch, prng_key, smi_eta):
    return elbo_smi(
        lambda_tuple=tuple(state.params for state in state_tuple),
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
        smi_eta=smi_eta,
    )

  if state_tuple[0].step < config.training_steps:
    logging.info('Training variational posterior...')

  # Reset random key sequence
  prng_seq = hk.PRNGSequence(config.seed)

  while state_tuple[0].step < config.training_steps:

    # Plots to monitor training
    if ((state_tuple[0].step == 0) or
        (state_tuple[0].step % config.log_img_steps == 0)):
      logging.info("\t\t Logging plots...")
      az_data = sample_q_as_az(
          lambda_tuple=tuple(x.params for x in state_tuple),
          dataset=train_ds,
          prng_key=next(prng_seq),
          flow_get_fn_nocut=flow_get_fn_nocut,
          flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
          flow_kwargs=config.flow_kwargs,
          sample_shape=(config.num_samples_plot,),
      )
      plot.posterior_plots(
          az_data=az_data,
          show_phi_trace=False,
          show_theta_trace=False,
          show_loglinear_scatter=True,
          show_theta_pairplot=True,
          eta=config.smi_eta_cancer,
          suffix=f"_eta_cancer_{float(config.smi_eta_cancer):.3f}",
          step=state_tuple[0].step,
          workdir_png=workdir,
          summary_writer=summary_writer,
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
          state_tuple=state_tuple,
          batch=train_ds,
          prng_key=next(prng_seq),
          smi_eta=smi_eta,
      )
      for k, v in elbo_dict.items():
        summary_writer.scalar(
            tag=f'elbo_{k}',
            value=v.mean(),
            step=state_tuple[0].step,
        )

    # Save checkpoints
    for state, mngr in zip(state_tuple, orbax_ckpt_mngrs):
      mngr.save(step=int(state.step), items=state)

  logging.info('Final training step: %i', state_tuple[0].step)

  # Last plot of posteriors
  az_data = sample_q_as_az(
      lambda_tuple=tuple(x.params for x in state_tuple),
      dataset=train_ds,
      prng_key=next(prng_seq),
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_plot,),
  )
  plot.posterior_plots(
      az_data=az_data,
      show_phi_trace=False,
      show_theta_trace=False,
      show_loglinear_scatter=True,
      show_theta_pairplot=True,
      eta=config.smi_eta_cancer,
      suffix=f"_eta_cancer_{float(config.smi_eta_cancer):.3f}",
      step=state_tuple[0].step,
      workdir_png=workdir,
      summary_writer=summary_writer,
  )

  return state_tuple


# # For debugging
# config = get_config()
# eta = 1.000
# import pathlib
# workdir = str(pathlib.Path.home() / f'modularbayes-output-exp/epidemiology/nsf/eta_{eta:.3f}')
# config.smi_eta_cancer = eta
# # train_and_evaluate(config, workdir)
