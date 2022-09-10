"""A simple example of variational SMI on the Random Effects model."""
import pathlib

from absl import logging

import numpy as np

from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax
import distrax

import flows
import log_prob_fun
import plot

from modularbayes._src.utils.training import TrainState
from modularbayes import (flatten_dict, initial_state_ckpt, update_states,
                          save_checkpoint)
from modularbayes._src.typing import (Any, Array, Batch, ConfigDict, Dict,
                                      IntLike, List, Mapping, Optional, PRNGKey,
                                      Sequence, SmiEta, SummaryWriter, Tuple,
                                      Union)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def get_dataset(
    num_obs_groups: Array,
    loc_groups: Array,
    scale_groups: Array,
    prng_key: PRNGKey,
) -> Dict[str, Array]:
  """Generate random effects data as in Liu 2009."""

  num_groups = len(num_obs_groups)
  assert len(loc_groups) == num_groups
  assert len(scale_groups) == num_groups

  num_obs_groups = jnp.array(num_obs_groups).astype(int)
  loc_groups = jnp.array(loc_groups)
  scale_groups = jnp.array(scale_groups)

  loc_pointwise = jnp.repeat(loc_groups, num_obs_groups)
  scale_pointwise = jnp.repeat(scale_groups, num_obs_groups)

  z = jax.random.normal(key=prng_key, shape=loc_pointwise.shape)

  data = {
      'Y': loc_pointwise + z * scale_pointwise,
      'group': jnp.repeat(jnp.arange(num_groups), num_obs_groups),
      'num_obs_groups': num_obs_groups,
  }

  return data


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


def q_distr_sigma(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    sample_shape: Union[IntLike, Sequence[IntLike]],
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  # Define normalizing flows
  q_distr = getattr(flows, flow_name + '_sigma')(**flow_kwargs)

  # Sample from flow
  (sigma_sample, sigma_log_prob_posterior,
   sigma_base_sample) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=sample_shape,
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
      context=sigma_base_sample,
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
    sample_shape: Union[IntLike, Sequence[IntLike]],
) -> Dict[str, Any]:
  """Sample from model posterior"""

  prng_seq = hk.PRNGSequence(prng_key)

  # sigma
  q_distr_out = hk.transform(q_distr_sigma).apply(
      params_tuple[0],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=sample_shape,
  )

  # beta and tau
  q_distr_out_beta_tau = hk.transform(q_distr_beta_tau).apply(
      params_tuple[1],
      next(prng_seq),
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sigma_base_sample=q_distr_out['sigma_base_sample'],
      is_aux=False,
  )
  q_distr_out['posterior_sample'].update(
      q_distr_out_beta_tau['posterior_sample'])
  q_distr_out['beta_tau_log_prob'] = q_distr_out_beta_tau['beta_tau_log_prob']

  if flow_kwargs.is_smi:
    q_distr_out_beta_tau_aux = hk.transform(q_distr_beta_tau).apply(
        params_tuple[2],
        next(prng_seq),
        flow_name=flow_name,
        flow_kwargs=flow_kwargs,
        sigma_base_sample=q_distr_out['sigma_base_sample'],
        is_aux=True,
    )
    q_distr_out['posterior_sample'].update(
        q_distr_out_beta_tau_aux['posterior_sample'])
    q_distr_out['beta_tau_aux_log_prob'] = q_distr_out_beta_tau_aux[
        'beta_tau_aux_log_prob']

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
      'sigma',
  ]
  refit_params_names = [
      'beta',
      'tau',
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
        q_distr_out['sigma_log_prob'] + q_distr_out['beta_tau_aux_log_prob'])

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
        jax.lax.stop_gradient(q_distr_out['sigma_log_prob']) +
        q_distr_out['beta_tau_log_prob'])
  else:
    log_q_stg2 = (
        q_distr_out['sigma_log_prob'] + q_distr_out['beta_tau_log_prob'])

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
    config: ConfigDict,
    num_samples_plot: int,
    suffix: Optional[str] = None,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  # Sample from posterior
  q_distr_out = sample_all_flows(
      params_tuple=[state.params for state in state_list],
      prng_key=prng_key,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(num_samples_plot,),
  )

  plot.posterior_samples(
      posterior_sample_dict=q_distr_out['posterior_sample'],
      step=state_list[0].step,
      summary_writer=summary_writer,
      suffix=suffix,
      workdir_png=workdir_png,
  )


def compute_elpd(
    posterior_sample_dict: Dict[str, Any],
    batch: Batch,
    y_new: Optional[Array] = None,
) -> Mapping[str, Array]:
  """Compute ELPD.

  Estimates the ELPD baased on two Monte Carlo approximations:
    1) Using WAIC.
    2) Assuming we have a sample of new data from the true generative model.
      Compute the Expected Log-Predictive density over the new data.

  Args:
    posterior_sample_dict: Dictionary of posterior samples.
    batch: Batch of data (the one that was for training).
    y_new: New data from the true generative model (not used during training).

  Returns:
    Dictionary of ELPD estimates, with keys:
      - 'elpd_waic_pointwise': WAIC-based estimate.
      - 'elpd_mc_pointwise': Expected Log-Predictive density over the new data.
  """

  # Initialize dictionary for output
  elpd_out = {}

  num_samples, _ = posterior_sample_dict['beta'].shape

  ### WAIC ###

  # Posterior predictive distribution
  predictive_dist = distrax.Normal(
      loc=posterior_sample_dict['beta'][:, batch['group']],
      scale=posterior_sample_dict['sigma'][:, batch['group']],
  )

  # Compute LPD
  loglik_pointwise_insample = predictive_dist.log_prob(batch['Y'])
  lpd_pointwise = jax.scipy.special.logsumexp(
      loglik_pointwise_insample, axis=0) - jnp.log(num_samples)
  elpd_out['lpd_pointwise'] = lpd_pointwise

  # Estimated effective number of parameters
  # Variance over samples
  p_waic_pointwise = jnp.var(loglik_pointwise_insample, axis=0)
  # # Sum over observations
  # p_waic = p_waic.sum()

  elpd_waic_pointwise = lpd_pointwise - p_waic_pointwise

  elpd_out['elpd_waic_pointwise'] = elpd_waic_pointwise

  # Approximate ELPD using Monte Carlo sampling from the true generative process
  if y_new is not None:
    loglik_pointwise_outofsample = jax.vmap(predictive_dist.log_prob)(y_new)

    # Computed log-pointwise predictive density
    # Average over posterior samples
    elpd_mc_pointwise = jax.scipy.special.logsumexp(
        loglik_pointwise_outofsample, axis=1) - jnp.log(
            num_samples)  # colLogMeanExps
    # Average over generative model samples
    elpd_mc_pointwise = elpd_mc_pointwise.mean(0)

    elpd_out['elpd_mc_pointwise'] = elpd_mc_pointwise

  return elpd_out


compute_elpd_jit = jax.jit(compute_elpd)


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

  smi_eta = config.flow_kwargs.smi_eta

  # num_groups is also a parameter of the flow,
  # as it define its dimension
  config.flow_kwargs.num_groups = config.num_groups
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

  ### Initialize States ###
  # Here we use three different states defining three separate flow models:
  #   -sigma
  #   -beta and tau
  #   -auxiliary beta and tau

  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')
  state_list = []
  state_name_list = []

  state_name_list.append('sigma')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=hk.transform(q_distr_sigma),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'sample_shape': (config.num_samples_elbo,)
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
      sample_shape=(config.num_samples_elbo,),
  )['sigma_base_sample']

  state_name_list.append('beta_tau')
  state_list.append(
      initial_state_ckpt(
          checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
          forward_fn=hk.transform(q_distr_beta_tau),
          forward_fn_kwargs={
              'flow_name': config.flow_name,
              'flow_kwargs': config.flow_kwargs,
              'sigma_base_sample': sigma_base_sample_init,
              'is_aux': False,
          },
          prng_key=next(prng_seq),
          optimizer=make_optimizer(**config.optim_kwargs),
      ))
  if config.flow_kwargs.is_smi:
    state_name_list.append('beta_tau_aux')
    state_list.append(
        initial_state_ckpt(
            checkpoint_dir=f'{checkpoint_dir}/{state_name_list[-1]}',
            forward_fn=hk.transform(q_distr_beta_tau),
            forward_fn_kwargs={
                'flow_name': config.flow_name,
                'flow_kwargs': config.flow_kwargs,
                'sigma_base_sample': sigma_base_sample_init,
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
          sample_shape=(config.num_samples_eval,),
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

  while state_list[0].step < config.training_steps:

    # Plots to monitor during training
    if ((state_list[0].step == 0) or
        (state_list[0].step % config.log_img_steps == 0)):
      # print("Logging images...\n")
      log_images(
          state_list=state_list,
          prng_key=next(prng_seq),
          config=config,
          num_samples_plot=int(config.num_samples_plot),
          suffix=config.plot_suffix,
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

    # Metrics for evaluation
    if state_list[0].step % config.eval_steps == 0:

      logging.info("STEP: %5d; training loss: %.3f", state_list[0].step - 1,
                   metrics["train_loss"])

      # Compute ELPD
      q_distr_out_eval = sample_all_flows(
          params_tuple=[state.params for state in state_list],
          prng_key=next(prng_seq),
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          sample_shape=(config.num_samples_eval,),
      )
      elpd_dict = compute_elpd_jit(
          posterior_sample_dict=q_distr_out_eval['posterior_sample'],
          batch=train_ds,
          y_new=y_new,
      )

      # Add pointwise elpd and lpd across observations to metrics dictionary
      metrics['lpd'] = elpd_dict['lpd_pointwise'].sum()
      metrics['elpd_mc'] = elpd_dict['elpd_mc_pointwise'].sum()
      metrics['elpd_waic'] = elpd_dict['elpd_waic_pointwise'].sum()

      # Log metrics to tensorboard
      metrics_to_log = ['lpd', 'elpd_mc', 'elpd_waic']
      for metric in metrics_to_log:
        summary_writer.scalar(
            tag=metric,
            value=metrics[metric],
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
      num_samples_plot=int(config.num_samples_plot),
      suffix=config.plot_suffix,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )

  return state_list
