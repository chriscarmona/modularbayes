"""A simple example of a flow model trained on Epidemiology data."""
import pathlib

from absl import logging

import numpy as np
from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax

import flows
import log_prob_fun
import plot
import data

from modularbayes._src.utils.training import TrainState
from modularbayes import (flatten_dict, initial_state_ckpt, update_states,
                          save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict,
                                      IntLike, List, Optional, PRNGKey,
                                      Sequence, SmiEta, SummaryWriter, Tuple,
                                      Union)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def load_dataset() -> Dict[str, Array]:
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


def q_distr_phi(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    sample_shape: Union[IntLike, Sequence[IntLike]],
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  # Define normalizing flows
  q_distr = getattr(flows, flow_name + '_phi')(**flow_kwargs)

  # Sample from flows
  (phi_sample, phi_log_prob_posterior,
   phi_base_sample) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=sample_shape,
   )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows.split_flow_phi(
          samples=phi_sample,
          **flow_kwargs,
      ))

  # sample from base distribution that generated phi
  q_distr_out['phi_base_sample'] = phi_base_sample

  # log P(phi)
  q_distr_out['phi_log_prob'] = phi_log_prob_posterior

  return q_distr_out


def q_distr_theta(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    phi_base_sample: Array,
    is_aux: bool,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  num_samples = phi_base_sample.shape[0]

  # Define normalizing flows
  q_distr = getattr(flows, flow_name + '_theta')(**flow_kwargs)

  # Sample from flows
  (theta_sample, theta_log_prob_posterior) = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=(num_samples,),
      context=phi_base_sample,
  )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows.split_flow_theta(
          samples=theta_sample,
          is_aux=is_aux,
          **flow_kwargs,
      ))

  # log P(theta|phi)
  q_distr_out['theta_' + ('aux_' if is_aux else '') +
              'log_prob'] = theta_log_prob_posterior

  return q_distr_out


def sample_all_flows(
    params_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    sample_shape: Union[IntLike, Sequence[IntLike]],
) -> Dict[str, Any]:
  """Sample from model posterior"""

  prng_seq = hk.PRNGSequence(prng_key)

  # phi
  q_distr_out = hk.transform(q_distr_phi).apply(
      params_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=sample_shape,
  )

  # theta
  q_distr_out_theta = hk.transform(q_distr_theta).apply(
      params_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      phi_base_sample=q_distr_out['phi_base_sample'],
      is_aux=False,
  )
  q_distr_out['posterior_sample'].update(q_distr_out_theta['posterior_sample'])
  q_distr_out['theta_log_prob'] = q_distr_out_theta['theta_log_prob']

  if flow_kwargs.is_smi:
    q_distr_out_theta_aux = hk.transform(q_distr_theta).apply(
        params_tuple[2],
        next(prng_seq),
        flow_name=flow_name,
        flow_kwargs=flow_kwargs,
        phi_base_sample=q_distr_out['phi_base_sample'],
        is_aux=True,
    )
    q_distr_out['posterior_sample'].update(
        q_distr_out_theta_aux['posterior_sample'])
    q_distr_out['theta_aux_log_prob'] = q_distr_out_theta_aux[
        'theta_aux_log_prob']

  return q_distr_out


def elbo_estimate(
    params_tuple: Tuple[hk.Params],
    batch: Batch,
    prng_key: PRNGKey,
    num_samples: int,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    smi_eta: Optional[SmiEta],
) -> Dict[str, Array]:
  """Estimate ELBO

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=params_tuple,
      prng_key=prng_key,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples,),
  )

  is_smi = False if smi_eta is None else True

  shared_params_names = [
      'phi',
  ]
  refit_params_names = [
      'theta',
  ]

  # ELBO stage 1: Power posterior
  if is_smi:
    posterior_sample_dict_stg1 = {}
    for key in shared_params_names:
      posterior_sample_dict_stg1[key] = q_distr_out['posterior_sample'][key]
    for key in refit_params_names:
      posterior_sample_dict_stg1[key] = q_distr_out['posterior_sample'][key +
                                                                        '_aux']
    log_prob_joint_stg1 = log_prob_fun.log_prob_joint(
        batch=batch,
        posterior_sample_dict=posterior_sample_dict_stg1,
        smi_eta=smi_eta,
    )
    log_q_stg1 = (
        q_distr_out['phi_log_prob'] + q_distr_out['theta_aux_log_prob'])

    elbo_stg1 = log_prob_joint_stg1 - log_q_stg1
  else:
    elbo_stg1 = 0.

  # ELBO stage 2: Refit theta
  posterior_sample_dict_stg2 = {}
  for key in shared_params_names:
    if is_smi:
      posterior_sample_dict_stg2[key] = jax.lax.stop_gradient(
          q_distr_out['posterior_sample'][key])
    else:
      posterior_sample_dict_stg2[key] = q_distr_out['posterior_sample'][key]
  for key in refit_params_names:
    posterior_sample_dict_stg2[key] = q_distr_out['posterior_sample'][key]
  log_prob_joint_stg2 = log_prob_fun.log_prob_joint(
      batch=batch,
      posterior_sample_dict=posterior_sample_dict_stg2,
      smi_eta=None,
  )
  if is_smi:
    log_q_stg2 = (
        jax.lax.stop_gradient(q_distr_out['phi_log_prob']) +
        q_distr_out['theta_log_prob'])
  else:
    log_q_stg2 = (q_distr_out['phi_log_prob'] + q_distr_out['theta_log_prob'])

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


def log_images(
    state_list: List[TrainState],
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    num_samples_plot: int,
    smi_eta: Optional[SmiEta] = None,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      prng_key=prng_key,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples_plot,),
  )

  plot.posterior_samples(
      posterior_sample_dict=q_distr_out['posterior_sample'],
      summary_writer=summary_writer,
      step=state_list[0].step,
      eta=smi_eta['modules'][0][1],
      workdir_png=workdir_png,
  )


def compute_elpd(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    num_samples: int,
):
  """Compute ELPD via WAIC"""

  # Sample from flow
  q_distr_out = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      prng_key=prng_key,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples,),
  )

  loglik_pointwise = log_prob_fun.log_lik_vectorised(
      batch['Z'],
      batch['Y'],
      batch['N'],
      batch['T'],
      q_distr_out['posterior_sample']['phi'],
      q_distr_out['posterior_sample']['theta'],
  )

  lpd = jax.scipy.special.logsumexp(
      loglik_pointwise, axis=0) - jnp.log(num_samples)  # colLogMeanExps
  p_waic = jnp.var(loglik_pointwise, axis=0)
  elpd_waic = lpd - p_waic

  # posterior_az = az.InferenceData(
  #     posterior=az.dict_to_dataset({
  #         k: np.expand_dims(v, axis=0)
  #         for k, v in posterior_sample_dict.items()
  #     }),
  #     log_likelihood=az.dict_to_dataset(
  #         {'x': np.expand_dims(loglik_pointwise[:, :, 1], axis=0)}),
  # )
  # posterior_az.log_likelihood.stack(__sample__=("chain", "draw")).shape
  # elpd_data = az.loo(data=posterior_az, pointwise=True)

  return elpd_waic


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

  smi_eta = {'modules': [[1.0, config.smi_eta]]}

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

  if smi_eta is not None:
    smi_eta = {k: jnp.array(v) for k, v in smi_eta.items()}

  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')
  state_list = []
  state_name_list = []

  state_name_list.append('phi')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=hk.transform(q_distr_phi),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'sample_shape': (config.num_samples_elbo,)
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))

  # Get an initial sample of phi
  # (used below to initialize theta)
  phi_base_sample_init = hk.transform(q_distr_phi).apply(
      state_list[0].params,
      next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_elbo,),
  )['phi_base_sample']

  state_name_list.append('theta')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=hk.transform(q_distr_theta),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'phi_base_sample': phi_base_sample_init,
              'is_aux': False,
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))
  if config.flow_kwargs.is_smi:
    state_name_list.append('theta_aux')
    state_list.append(
        initial_state_ckpt(
            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
            forward_fn=hk.transform(q_distr_theta),
            forward_fn_kwargs={
                'flow_name': config.flow_name,
                'flow_kwargs': config.flow_kwargs,
                'phi_base_sample': phi_base_sample_init,
                'is_aux': True,
            },
            prng_key=next(prng_seq),
            optimizer=make_optimizer(**config.optim_kwargs),
        ))

  # Print a useful summary of the execution of the flow architecture.
  logging.info('FLOW PHI:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: hk.transform(q_distr_phi).apply(
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

  logging.info('FLOW THETA:')
  tabulate_fn_ = hk.experimental.tabulate(
      f=lambda params, prng_key: hk.transform(q_distr_theta).apply(
          params,
          prng_key,
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          phi_base_sample=phi_base_sample_init,
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
  update_states_jit = lambda state_list, batch, prng_key, smi_eta: update_states(
      state_list=state_list,
      batch=batch,
      prng_key=prng_key,
      optimizer=make_optimizer(**config.optim_kwargs),
      loss_fn=loss,
      loss_fn_kwargs={
          'num_samples': config.num_samples_elbo,
          'flow_name': config.flow_name,
          'flow_kwargs': config.flow_kwargs,
          'smi_eta': smi_eta,
      },
  )
  update_states_jit = jax.jit(update_states_jit)

  elbo_validation_jit = lambda state_list, batch, prng_key, smi_eta: elbo_estimate(
      params_tuple=[state.params for state in state_list],
      batch=batch,
      prng_key=prng_key,
      num_samples=config.num_samples_eval,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      smi_eta=smi_eta,
  )
  elbo_validation_jit = jax.jit(elbo_validation_jit)

  if state_list[0].step < config.training_steps:
    logging.info('Training variational posterior...')

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
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          num_samples_plot=config.num_samples_plot,
          smi_eta=smi_eta,
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
  log_images(
      state_list=state_list,
      prng_key=next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      num_samples_plot=config.num_samples_plot,
      smi_eta=smi_eta,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )

  return state_list


# # For debugging
# config = get_config()
# config.flow_kwargs.smi_eta = {'modules': [[1.0, 1.0]]}
# workdir = pathlib.Path.home()/'smi/output/epidemiology/mean_field/eta_1.0'
# train_and_evaluate(config, workdir)