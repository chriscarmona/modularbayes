"""Training a Variational Meta-Posterior, using the VMP flow."""

import pathlib

from absl import logging

import numpy as np

from arviz import InferenceData

from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax

import flows
from flows import split_flow_nocut, split_flow_cut
from log_prob_fun import (ModelParams, ModelParamsCut, SmiEta, logprob_joint,
                          sample_eta_values)
import plot
from train_flow import load_data, make_optimizer, sample_q_as_az

from modularbayes import (sample_q_nocut, sample_q_cutgivennocut,
                          elbo_smi_vmpflow)
from modularbayes._src.utils.training import TrainState
from modularbayes import (flatten_dict, initial_state_ckpt, update_states,
                          save_checkpoint)
from modularbayes._src.typing import (Any, Array, Callable, ConfigDict, Dict,
                                      List, Optional, PRNGKey, Tuple)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def loss(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  ### Compute ELBO ###
  elbo_dict = elbo_smi_vmpflow(lambda_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2']))

  return loss_avg


def log_images(
    state_list: List[TrainState],
    prng_key: PRNGKey,
    config: ConfigDict,
    dataset: Dict[str, Any],
    num_samples_plot: int,
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    summary_writer: Optional[tensorboard.SummaryWriter] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

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

  state_name_list.append('lambda_nocut')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=sample_q_nocut,
          forward_fn_kwargs={
              'flow_get_fn':
                  flow_get_fn_nocut,
              'flow_kwargs':
                  config.flow_kwargs,
              'split_flow_fn':
                  split_flow_nocut,
              'eta_values':
                  jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # Get an initial sample of the base distribution of nocut params
  nocut_base_sample_init = sample_q_nocut.apply(
      state_list[0].params,
      next(prng_seq),
      flow_get_fn=flow_get_fn_nocut,
      flow_kwargs=config.flow_kwargs,
      split_flow_fn=split_flow_nocut,
      eta_values=jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
  )['sample_base']

  state_name_list.append('lambda_cut')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=sample_q_cutgivennocut,
          forward_fn_kwargs={
              'flow_get_fn':
                  flow_get_fn_cutgivennocut,
              'flow_kwargs':
                  config.flow_kwargs,
              'split_flow_fn':
                  split_flow_cut,
              'nocut_base_sample':
                  nocut_base_sample_init,
              'eta_values':
                  jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))
  if config.flow_kwargs.is_smi:
    state_name_list.append('lambda_cut_aux')
    state_list.append(
        initial_state_ckpt(
            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
            forward_fn=sample_q_cutgivennocut,
            forward_fn_kwargs={
                'flow_get_fn':
                    flow_get_fn_cutgivennocut,
                'flow_kwargs':
                    config.flow_kwargs,
                'split_flow_fn':
                    split_flow_cut,
                'nocut_base_sample':
                    nocut_base_sample_init,
                'eta_values':
                    jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
            },
            prng_key=next(prng_seq),
            optimizer=make_optimizer(**config.optim_kwargs),
        ))

  # Print a useful summary of the execution of the flow architecture.
  logging.info('\nFlow no-cut parameters:\n')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: sample_q_nocut.apply(
          params,
          prng_key,
          flow_get_fn=flow_get_fn_nocut,
          flow_kwargs=config.flow_kwargs,
          split_flow_fn=split_flow_nocut,
          eta_values=jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
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

  logging.info('\nFlow cut parameters:\n')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: sample_q_cutgivennocut.apply(
          params,
          prng_key,
          flow_get_fn=flow_get_fn_cutgivennocut,
          flow_kwargs=config.flow_kwargs,
          split_flow_fn=split_flow_cut,
          nocut_base_sample=nocut_base_sample_init,
          eta_values=jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
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

  if state_list[0].step < config.training_steps:
    logging.info('Training Variational Meta-Posterior (VMP-flow)...')

  # Reset random key sequence
  prng_seq = hk.PRNGSequence(config.seed)

  while state_list[0].step < config.training_steps:

    # Plots to monitor training
    if ((state_list[0].step == 0) or
        (state_list[0].step % config.log_img_steps == 0)):
      # print("Logging images...\n")
      log_images(
          state_list=state_list,
          prng_key=next(prng_seq),
          config=config,
          dataset=train_ds,
          num_samples_plot=config.num_samples_plot,
          flow_get_fn_nocut=flow_get_fn_nocut,
          flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
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

      elbo_dict = elbo_validation_jit(
          state_list=state_list,
          batch=train_ds,
          prng_key=next(prng_seq),
      )
      for k, v in elbo_dict.items():
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

  # Saving checkpoint at the end of the training process
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
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )

  return state_list


# # For debugging
# config = get_config()
# import pathlib
# workdir = str(pathlib.Path.home() / f'modularbayes-output-exp/random_effects/nsf/vmp_flow')
# # train_and_evaluate(config, workdir)
