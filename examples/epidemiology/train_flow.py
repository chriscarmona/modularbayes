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
import data

from modularbayes._src.utils.training import TrainState
from modularbayes import (flatten_dict, initial_state_ckpt, update_states,
                          save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict,
                                      IntLike, List, Optional, PRNGKey,
                                      Sequence, Tuple, Union)

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


def sample_q_nocut(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    sample_shape: Union[IntLike, Sequence[IntLike]],
) -> Dict[str, Any]:
  """Sample from variational posterior for no-cut parameters."""

  q_output = {}

  # Define normalizing flows
  q_distr = getattr(flows, 'get_q_nocut_' + flow_name)(**flow_kwargs)

  # Sample from flows
  (sample_flow_concat, sample_logprob,
   sample_base) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=sample_shape,
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
) -> Dict[str, Any]:
  """Sample from variational posterior for cut parameters
  Conditional on values of no-cut parameters."""

  q_output = {}

  num_samples = nocut_base_sample.shape[0]

  # Define normalizing flows
  q_distr = getattr(flows, 'get_q_cutgivennocut_' + flow_name)(**flow_kwargs)

  # Sample from flows
  (sample, sample_logprob) = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
      context=nocut_base_sample,
  )

  # Split flow into model parameters
  q_output['sample'] = jax.vmap(lambda x: flows.split_flow_cut(
      concat_params=x,
      **flow_kwargs,
  ))(
      sample)

  # log P(theta|phi)
  q_output['sample_logprob'] = sample_logprob

  return q_output


def sample_q(
    lambda_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    sample_shape: Union[IntLike, Sequence[IntLike]],
) -> Dict[str, Any]:
  """Sample from model posterior"""

  prng_seq = hk.PRNGSequence(prng_key)

  q_output = {}

  # Sample from q(no_cut_params)
  q_output_nocut_ = hk.transform(sample_q_nocut).apply(
      lambda_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=sample_shape,
  )

  # Sample from q(cut_params|no_cut_params)
  q_output_cut_ = hk.transform(sample_q_cutgivennocut).apply(
      lambda_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      nocut_base_sample=q_output_nocut_['sample_base'],
  )

  q_output['model_params_sample'] = ModelParams(**{
      **q_output_nocut_['sample']._asdict(),
      **q_output_cut_['sample']._asdict(),
  })
  q_output['log_q_nocut'] = q_output_nocut_['sample_logprob']
  q_output['log_q_cut'] = q_output_cut_['sample_logprob']

  if flow_kwargs.is_smi:
    q_output_cut_aux_ = hk.transform(sample_q_cutgivennocut).apply(
        lambda_tuple[2],
        next(prng_seq),
        flow_name=flow_name,
        flow_kwargs=flow_kwargs,
        nocut_base_sample=q_output_nocut_['sample_base'],
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
    smi_eta: Optional[SmiEta] = None,
) -> Dict[str, Array]:
  """Estimate ELBO

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  # Sample from flow
  q_distr_out = sample_q(
      lambda_tuple=params_tuple,
      prng_key=prng_key,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples,),
  )

  is_smi = False if smi_eta is None else True

  # ELBO stage 1: Power posterior
  if is_smi:
    log_prob_joint_stg1 = jax.vmap(lambda x: log_prob_fun.logprob_joint(
        batch=batch,
        model_params=x,
        prior_hparams=prior_hparams,
        smi_eta=smi_eta,
    ))(
        q_distr_out['model_params_aux_sample'])
    log_q_stg1 = (q_distr_out['log_q_nocut'] + q_distr_out['log_q_cut_aux'])

    elbo_stg1 = log_prob_joint_stg1 - log_q_stg1
  else:
    elbo_stg1 = 0.

  # ELBO stage 2: Conventional posterior (with stop_gradient)
  if is_smi:
    model_params_stg2 = ModelParams(
        **{
            k: (v if k in ModelParamsCut._fields else jax.lax.stop_gradient(v))
            for k, v in q_distr_out['model_params_sample']._asdict().items()
        })
  else:
    model_params_stg2 = q_distr_out['model_params_sample']

  log_prob_joint_stg2 = jax.vmap(lambda x: log_prob_fun.logprob_joint(
      batch=batch,
      model_params=x,
      prior_hparams=prior_hparams,
      smi_eta=None,
  ))(
      model_params_stg2)
  if is_smi:
    log_q_stg2 = (
        jax.lax.stop_gradient(q_distr_out['log_q_nocut']) +
        q_distr_out['log_q_cut'])
  else:
    log_q_stg2 = (q_distr_out['log_q_nocut'] + q_distr_out['log_q_cut'])

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
    num_samples: int,
) -> InferenceData:
  """Plots to monitor during training."""
  # Sample from flow
  q_distr_out = sample_q(
      lambda_tuple=[state.params for state in state_list],
      prng_key=prng_key,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples,),
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
              'sample_shape': (config.num_samples_elbo,)
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # Get an initial sample of the base distribution of nocut params
  # (used below to initialize theta)
  nocut_base_sample_init = hk.transform(sample_q_nocut).apply(
      state_list[0].params,
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_elbo,),
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
          sample_shape=(config.num_samples_elbo,),
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
  def update_states_jit(state_list, batch, prng_key, smi_eta):
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
            'smi_eta': smi_eta,
        },
    )

  @jax.jit
  def elbo_validation_jit(state_list, batch, prng_key, smi_eta):
    return elbo_estimate(
        params_tuple=[state.params for state in state_list],
        batch=batch,
        prng_key=prng_key,
        num_samples=config.num_samples_eval,
        flow_name=config.flow_name,
        flow_kwargs=config.flow_kwargs,
        prior_hparams=config.prior_hparams,
        smi_eta=smi_eta,
    )

  if state_list[0].step < config.training_steps:
    logging.info('Training variational posterior...')

  # Reset random key sequence
  prng_seq = hk.PRNGSequence(config.seed)

  while state_list[0].step < config.training_steps:

    # Plots to monitor training
    if ((state_list[0].step == 0) or
        (state_list[0].step % config.log_img_steps == 0)):
      # print("Logging images...\n")
      hpv_az = sample_q_as_az(
          state_list=state_list,
          hpv_dataset=train_ds,
          prng_key=next(prng_seq),
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          num_samples=config.num_samples_plot,
      )
      plot.hpv_plots_arviz(
          hpv_az=hpv_az,
          show_phi_trace=False,
          show_theta_trace=False,
          show_loglinear_scatter=True,
          show_theta_pairplot=True,
          eta=config.smi_eta_cancer,
          suffix=f"_eta_cancer_{float(config.smi_eta_cancer):.3f}",
          step=state_list[0].step,
          workdir_png=workdir,
          summary_writer=summary_writer,
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
        smi_eta=smi_eta,
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
          smi_eta=smi_eta,
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
  hpv_az = sample_q_as_az(
      state_list=state_list,
      hpv_dataset=train_ds,
      prng_key=next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      num_samples=config.num_samples_plot,
  )
  plot.hpv_plots_arviz(
      hpv_az=hpv_az,
      show_phi_trace=False,
      show_theta_trace=False,
      show_loglinear_scatter=True,
      show_theta_pairplot=True,
      eta=config.smi_eta_cancer,
      suffix=f"_eta_cancer_{float(config.smi_eta_cancer):.3f}",
      step=state_list[0].step,
      workdir_png=workdir,
      summary_writer=summary_writer,
  )

  return state_list


# # For debugging
# config = get_config()
# eta = 1.000
# import pathlib
# workdir = str(pathlib.Path.home() / f'modularbayes-output-exp/epidemiology/nsf/eta_{eta:.3f}')
# config.smi_eta_cancer = eta
# # train_and_evaluate(config, workdir)