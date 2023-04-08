"""A simple example of a flow model trained on Epidemiology data."""
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
import log_prob_fun
from log_prob_fun import ModelParams, ModelParamsCut, SmiEta
import plot
from train_flow import load_data, make_optimizer

from modularbayes._src.utils.training import TrainState
from modularbayes import (flatten_dict, initial_state_ckpt, update_states,
                          save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict, List,
                                      Optional, PRNGKey, SummaryWriter, Tuple)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def sample_q_nocut(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    eta_values: Array,
) -> Dict[str, Any]:
  """Sample from variational posterior for no-cut parameters."""

  q_output = {}

  # Define normalizing flows
  q_distr = getattr(flows, 'get_q_nocut_' + flow_name)(**flow_kwargs)

  num_samples = eta_values.shape[0]

  # Sample from flows
  (sample_flow_concat, sample_logprob,
   sample_base) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=(num_samples,),
       context=(eta_values, None),
   )

  # Split flow into model parameters
  q_output['sample'] = jax.vmap(lambda x: flows.split_flow_nocut(
      concat_params=x,
      **flow_kwargs,
  ))(
      sample_flow_concat)

  # sample from base distribution that generated phi
  q_output['sample_base'] = sample_base

  # variational posterior evaluated in the sample
  q_output['sample_logprob'] = sample_logprob

  return q_output


def sample_q_cutgivennocut(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    nocut_base_sample: Array,
    eta_values: Array,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_output = {}

  num_samples = nocut_base_sample.shape[0]

  # Define normalizing flows
  q_distr = getattr(flows, 'get_q_cut_' + flow_name)(**flow_kwargs)

  # Sample from flow
  (sample_flow_concat, sample_logprob) = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
      context=(eta_values, nocut_base_sample),
  )

  # Split flow into model parameters
  q_output['sample'] = jax.vmap(lambda x: flows.split_flow_cut(
      concat_params=x,
      **flow_kwargs,
  ))(
      sample_flow_concat)

  # log P(theta|phi)
  q_output['sample_logprob'] = sample_logprob

  return q_output


def sample_q(
    lambda_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    eta_values: Array,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  prng_seq = hk.PRNGSequence(prng_key)

  q_output = {}
  # Sample from q_{lambda0}(no_cut_params)
  q_output_nocut_ = hk.transform(sample_q_nocut).apply(
      lambda_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      eta_values=eta_values,
  )

  # Sample from q_{lambda1}(cut_params|no_cut_params)
  q_output_cut_ = hk.transform(sample_q_cutgivennocut).apply(
      lambda_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      nocut_base_sample=q_output_nocut_['sample_base'],
      eta_values=eta_values,
  )

  q_output['model_params_sample'] = ModelParams(**{
      **q_output_nocut_['sample']._asdict(),
      **q_output_cut_['sample']._asdict(),
  })
  q_output['log_q_nocut'] = q_output_nocut_['sample_logprob']
  q_output['log_q_cut'] = q_output_cut_['sample_logprob']

  # Sample from q_{lambda2}(cut_params|no_cut_params)
  q_output_cut_aux_ = hk.transform(sample_q_cutgivennocut).apply(
      lambda_tuple[2],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      nocut_base_sample=q_output_nocut_['sample_base'],
      eta_values=eta_values,
  )
  q_output['model_params_aux_sample'] = ModelParams(
      **{
          **q_output_nocut_['sample']._asdict(),
          **q_output_cut_aux_['sample']._asdict(),
      })
  q_output['log_q_cut_aux'] = q_output_cut_aux_['sample_logprob']

  return q_output


def elbo_estimate(
    params_tuple: Tuple[hk.Params],
    batch: Batch,
    prng_key: PRNGKey,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    prior_hparams: Dict[str, float],
    sample_eta_kwargs: Dict[str, Any],
) -> Dict[str, Array]:
  """Estimate ELBO

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  prng_seq = hk.PRNGSequence(prng_key)

  # Sample eta values
  smi_etas = log_prob_fun.sample_eta_values(
      prng_key=next(prng_seq),
      num_samples=num_samples,
      **sample_eta_kwargs,
  )
  # Sample from flow
  q_distr_out = sample_q(
      lambda_tuple=params_tuple,
      prng_key=next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      eta_values=jnp.stack(smi_etas, axis=-1),
  )

  # ELBO stage 1: Power posterior
  log_prob_joint_stg1 = jax.vmap(lambda x, y: log_prob_fun.logprob_joint(
      batch=batch,
      model_params=x,
      prior_hparams=prior_hparams,
      smi_eta=y,
  ))(q_distr_out['model_params_aux_sample'], smi_etas)
  log_q_stg1 = (q_distr_out['log_q_nocut'] + q_distr_out['log_q_cut_aux'])
  elbo_stg1 = log_prob_joint_stg1 - log_q_stg1

  # ELBO stage 2: Conventional posterior (with stop_gradient)
  model_params_stg2 = ModelParams(
      **{
          k: (v if k in ModelParamsCut._fields else jax.lax.stop_gradient(v))
          for k, v in q_distr_out['model_params_sample']._asdict().items()
      })
  log_prob_joint_stg2 = jax.vmap(lambda x: log_prob_fun.logprob_joint(
      batch=batch,
      model_params=x,
      prior_hparams=prior_hparams,
      smi_eta=None,
  ))(
      model_params_stg2)
  log_q_stg2 = (
      jax.lax.stop_gradient(q_distr_out['log_q_nocut']) +
      q_distr_out['log_q_cut'])

  elbo_stg2 = log_prob_joint_stg2 - log_q_stg2

  elbo_dict = {'stage_1': elbo_stg1, 'stage_2': elbo_stg2}

  return elbo_dict


def loss(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  ### Compute ELBO ###
  elbo_dict = elbo_estimate(params_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2']))

  return loss_avg


def sample_q_as_az(
    state_list: List[TrainState],
    hpv_dataset: Dict[str, Any],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    eta_values: Array,
) -> InferenceData:
  """Plots to monitor during training."""
  # Sample from flow
  q_distr_out = sample_q(
      lambda_tuple=[state.params for state in state_list],
      prng_key=prng_key,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      eta_values=eta_values,
  )
  # Add dimension for "chains"
  model_params_az = jax.tree_map(lambda x: x[None, ...],
                                 q_distr_out['model_params_sample'])
  # Create InferenceData object
  hpv_az = plot.hpv_az_from_samples(
      hpv_dataset=hpv_dataset,
      model_params=model_params_az,
  )

  return hpv_az


def log_images(
    state_list: List[TrainState],
    prng_key: PRNGKey,
    config: ConfigDict,
    hpv_dataset: Dict[str, Any],
    num_samples_plot: int,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  assert len(config.smi_eta_cancer_plot) > 0, 'No eta values to plot'
  assert all((x >= 0.0) and (x <= 1.0)
             for x in config.smi_eta_cancer_plot), 'Invalid eta values'

  # Plot posterior samples
  for eta_cancer_i in config.smi_eta_cancer_plot:
    # Define eta with a single value
    smi_eta_plot = SmiEta(
        hpv=jnp.ones(num_samples_plot),
        cancer=eta_cancer_i * jnp.ones(num_samples_plot),
    )
    # Sample from flow
    hpv_az = sample_q_as_az(
        state_list=state_list,
        hpv_dataset=hpv_dataset,
        prng_key=next(prng_seq),
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        eta_values=jnp.stack(smi_eta_plot, axis=-1),
    )
    plot.hpv_plots_arviz(
        hpv_az=hpv_az,
        show_phi_trace=False,
        show_theta_trace=False,
        show_loglinear_scatter=True,
        show_theta_pairplot=True,
        eta=eta_cancer_i,
        suffix=f"_eta_cancer_{eta_cancer_i}",
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

  # Initialize random keys
  prng_seq = hk.PRNGSequence(config.seed)

  # Full dataset used everytime
  # Small data, no need to batch
  train_ds = load_data()

  phi_dim = train_ds['Z'].shape[0]
  theta_dim = 2

  # phi_dim and theta_dim are also arguments of the flow,
  # as they define its dimension
  config.flow_kwargs.phi_dim = phi_dim
  config.flow_kwargs.theta_dim = theta_dim
  # Also is_smi modifies the dimension of the flow, due to the duplicated params
  config.flow_kwargs.is_smi = True

  # writer = metric_writers.create_default_writer(
  #     logdir=workdir, just_logging=jax.host_id() != 0)
  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))
  else:
    summary_writer = None

  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')
  state_list = []
  state_name_list = []

  state_name_list.append('lambda_nocut')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=hk.transform(sample_q_nocut),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'eta_values': jnp.ones((config.num_samples_elbo, 2)),
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # Get an initial sample of phi
  # (used below to initialize theta)
  nocut_base_sample_init = hk.transform(sample_q_nocut).apply(
      state_list[0].params,
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      eta_values=jnp.ones((config.num_samples_elbo, 2)),
  )['sample_base']

  state_name_list.append('lambda_cut')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=hk.transform(sample_q_cutgivennocut),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'nocut_base_sample': nocut_base_sample_init,
              'eta_values': jnp.ones((config.num_samples_elbo, 2)),
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))
  if config.flow_kwargs.is_smi:
    state_name_list.append('lambda_cut_aux')
    state_list.append(
        initial_state_ckpt(
            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
            forward_fn=hk.transform(sample_q_cutgivennocut),
            forward_fn_kwargs={
                'flow_name': config.flow_name,
                'flow_kwargs': config.flow_kwargs,
                'nocut_base_sample': nocut_base_sample_init,
                'eta_values': jnp.ones((config.num_samples_elbo, 2)),
            },
            prng_key=next(prng_seq),
            optimizer=make_optimizer(**config.optim_kwargs),
        ))

  # Print a useful summary of the execution of the flow architecture.
  logging.info('\nFlow no-cut parameters:\n')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: hk.transform(sample_q_nocut).apply(
          params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          eta_values=jnp.ones((config.num_samples_elbo, 2)),
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
      f=lambda params, prng_key: hk.transform(sample_q_cutgivennocut).apply(
          params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          nocut_base_sample=nocut_base_sample_init,
          eta_values=jnp.ones((config.num_samples_elbo, 2)),
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
            'flow_name': config.flow_name,
            'flow_kwargs': config.flow_kwargs,
            'prior_hparams': config.prior_hparams,
            'sample_eta_kwargs': {
                'eta_sampling_a': config.eta_sampling_a,
                'eta_sampling_b': config.eta_sampling_b
            },
        },
    )

  @jax.jit
  def elbo_validation_jit(state_list, batch, prng_key):
    return elbo_estimate(
        params_tuple=tuple(state.params for state in state_list),
        batch=batch,
        prng_key=prng_key,
        num_samples=config.num_samples_eval,
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        prior_hparams=config.prior_hparams,
        sample_eta_kwargs={
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
          hpv_dataset=train_ds,
          num_samples_plot=config.num_samples_plot,
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
      hpv_dataset=train_ds,
      num_samples_plot=config.num_samples_plot,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )

  return state_list


# # For debugging
# config = get_config()
# import pathlib
# workdir = str(pathlib.Path.home() / f'modularbayes-output-exp/epidemiology/vmp_nsf')
# # train_and_evaluate(config, workdir)
