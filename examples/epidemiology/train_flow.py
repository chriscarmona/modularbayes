"""A simple example of a flow model trained on Epidemiology data."""
import pathlib

from absl import logging

import numpy as np

# from clu import metric_writers
from flax.metrics import tensorboard

import jax
from jax import numpy as jnp

import haiku as hk
import optax

import flows
import log_prob_fun
import plot

import modularbayes
from modularbayes import utils
from modularbayes.utils.training import TrainState
from modularbayes.typing import (Any, Array, Batch, ConfigDict, Dict, IntLike,
                                 Optional, PRNGKey, Sequence, SmiEta,
                                 SummaryWriter, Union)

# Set high precision for matrix multiplication in jax
jax.config.update('jax_default_matmul_precision', 'float32')

np.set_printoptions(suppress=True, precision=4)


def load_dataset() -> Dict[str, Array]:
  """Load Epidemiology data from Plummer."""
  data = dict(
      zip(['Z', 'N', 'Y', 'T'],
          np.split(modularbayes.data.epidemiology.to_numpy(), 4, axis=-1)))
  data = {key: jnp.array(value.squeeze()) for key, value in data.items()}
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


def q_distr(
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    sample_shape: Union[IntLike, Sequence[IntLike]],
) -> Dict[str, Any]:
  """Sample from model posterior"""

  q_distr_out = {}

  # Define normalizing flows
  q_distr_phi = getattr(flows, flow_name + '_phi')(**flow_kwargs)
  q_distr_theta = getattr(flows, flow_name + '_theta')(**flow_kwargs)

  # Sample from flows
  (phi_sample, phi_log_prob_posterior,
   phi_base_sample) = q_distr_phi.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(),
       sample_shape=sample_shape,
   )
  (theta_sample, theta_log_prob_posterior) = q_distr_theta.sample_and_log_prob(
      seed=hk.next_rng_key(),
      sample_shape=sample_shape,
      context=phi_base_sample,
  )

  # Split flow into model parameters
  q_distr_out['posterior_sample'] = {}
  q_distr_out['posterior_sample'].update(
      flows.split_flow_phi(
          samples=phi_sample,
          **flow_kwargs,
      ))
  q_distr_out['posterior_sample'].update(
      flows.split_flow_theta(
          samples=theta_sample,
          is_aux=False,
          **flow_kwargs,
      ))

  # log P(phi)
  q_distr_out['phi_log_prob'] = phi_log_prob_posterior
  # log P(theta|phi)
  q_distr_out['theta_log_prob'] = theta_log_prob_posterior

  if flow_kwargs.is_smi:
    # Repeat for auxiliary theta
    q_distr_theta_aux = getattr(flows, flow_name + '_theta')(**flow_kwargs)
    (theta_aux_sample,
     theta_aux_log_prob_posterior) = q_distr_theta_aux.sample_and_log_prob(
         seed=hk.next_rng_key(),
         sample_shape=sample_shape,
         context=phi_base_sample,
     )
    q_distr_out['posterior_sample'].update(
        flows.split_flow_theta(
            samples=theta_aux_sample,
            is_aux=True,
            **flow_kwargs,
        ))
    # log P(theta_aux|phi)
    q_distr_out['theta_aux_log_prob'] = theta_aux_log_prob_posterior

  return q_distr_out


def elbo_fn(
    q_distr_out: Dict[str, Any],
    smi_eta: Optional[SmiEta],
    batch: Batch,
) -> Dict[str, Array]:

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


def loss(
    params: hk.Params,
    batch: Batch,
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    num_samples_flow: int,
    smi_eta: Optional[SmiEta] = None,
) -> Array:
  """Define training loss function."""

  # Sample from flow
  q_distr_out = hk.transform(q_distr).apply(
      params,
      prng_key,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples_flow,),
  )

  ### Compute ELBO ###
  elbo_dict = elbo_fn(
      q_distr_out=q_distr_out,
      smi_eta=smi_eta,
      batch=batch,
  )

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict['stage_1'] + elbo_dict['stage_2']))

  return loss_avg


def log_images(
    state: TrainState,
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    num_samples_plot: int,
    smi_eta: Optional[SmiEta] = None,
    summary_writer: Optional[SummaryWriter] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""
  # Sample from posterior
  q_distr_out = hk.transform(q_distr).apply(
      state.params,
      prng_key,
      flow_name=flow_name,
      flow_kwargs=flow_kwargs,
      sample_shape=(num_samples_plot,),
  )

  plot.posterior_samples(
      posterior_sample_dict=q_distr_out['posterior_sample'],
      summary_writer=summary_writer,
      step=state.step,
      eta=smi_eta['modules'][0][1],
      workdir_png=workdir_png,
  )


def compute_elpd(
    state: TrainState,
    batch: Batch,
    prng_key: PRNGKey,
    flow_name: str,
    flow_kwargs: Dict[str, Any],
    num_samples: int,
):
  """Compute ELPD via WAIC"""

  # Sample from posterior
  q_distr_out = hk.transform(q_distr).apply(
      state.params,
      prng_key,
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

  smi_eta = config.flow_kwargs.smi_eta

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
    summary_writer.hparams(utils.flatten_dict(config))
  else:
    summary_writer = None

  if smi_eta is not None:
    smi_eta = {k: jnp.array(v) for k, v in smi_eta.items()}

  checkpoint_dir = str(pathlib.Path(workdir) / 'checkpoints')
  state = utils.initial_state_ckpt(
      checkpoint_dir=checkpoint_dir,
      forward_fn=hk.transform(q_distr),
      forward_fn_kwargs={
          'flow_name': config.flow_name,
          'flow_kwargs': config.flow_kwargs,
          'sample_shape': (config.num_samples_elbo,)
      },
      prng_key=next(prng_seq),
      optimizer=make_optimizer(**config.optim_kwargs),
  )

  # Print a useful summary of the execution of the flow architecture.
  _fwd_fn = lambda params, prng_key: hk.transform(q_distr).apply(
      params,
      prng_key,
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      sample_shape=(config.num_samples_elbo,),
  )
  summary = hk.experimental.tabulate(
      _fwd_fn,
      columns=(
          "module",
          "owned_params",
          "params_size",
          "params_bytes",
      ),
      filters=("has_params",),
  )(state.params, next(prng_seq))
  for line in summary.split("\n"):
    logging.info(line)

  update_state_jit = lambda state, batch, prng_key, smi_eta: utils.update_state(
      state=state,
      batch=batch,
      prng_key=prng_key,
      optimizer=make_optimizer(**config.optim_kwargs),
      loss_fn=loss,
      loss_fn_kwargs={
          'flow_name': config.flow_name,
          'flow_kwargs': config.flow_kwargs,
          'num_samples_flow': config.num_samples_elbo,
          'smi_eta': smi_eta,
      },
  )
  # update_state_jit = update_state_jit
  update_state_jit = jax.jit(update_state_jit)

  elpd_jitted = lambda state, batch, prng_key: compute_elpd(
      state,
      batch,
      prng_key,
      config.flow_name,
      config.flow_kwargs,
      int(config.num_samples_eval),
  )
  elpd_jitted = jax.jit(elpd_jitted)

  if state.step < config.training_steps:
    logging.info('Training variational posterior...')

  while state.step < config.training_steps:

    # Plots to monitor training
    if ((state.step == 0) or ((state.step > config.random_eta_steps) and
                              (state.step % config.log_img_steps == 0))):
      # print("Logging images...\n")
      log_images(
          state=state,
          prng_key=next(prng_seq),
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          num_samples_plot=config.num_samples_plot,
          smi_eta=smi_eta,
          summary_writer=summary_writer,
          workdir_png=workdir,
      )

    # Take randomn values of eta at the beginning of training
    if (smi_eta is not None) and (state.step < int(config.random_eta_steps)):
      smi_eta_step = smi_eta.copy()
      smi_eta_step['modules'] = jax.random.beta(
          key=next(prng_seq),
          a=0.2,
          b=0.2,
          shape=smi_eta['modules'].shape,
      )
    else:
      smi_eta_step = smi_eta

    # Log learning rate
    summary_writer.scalar(
        tag='learning_rate',
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(state.step),
        step=state.step,
    )

    # step = 0
    state, metrics = update_state_jit(
        state=state,
        batch=train_ds,
        prng_key=next(prng_seq),
        smi_eta=smi_eta_step,
    )
    # The computed training loss would correspond to the model before update
    summary_writer.scalar(
        tag='train_loss',
        value=metrics['train_loss'],
        step=state.step - 1,
    )

    # elpd = elpd_jitted(
    #     state,
    #     train_ds,
    #     next(prng_seq),
    # )

    if state.step == 1:
      logging.info("STEP: %5d; training loss: %.3f", state.step - 1,
                   metrics["train_loss"])

    if state.step % config.eval_steps == 0:
      logging.info("STEP: %5d; training loss: %.3f", state.step - 1,
                   metrics["train_loss"])

      # Sample from flow
      q_distr_out_eval = hk.transform(q_distr).apply(
          state.params,
          next(prng_seq),
          flow_name=config.flow_name,
          flow_kwargs=config.flow_kwargs,
          sample_shape=(config.num_samples_eval,),
      )

      # Compute ELBO
      elbo_dict_eval = elbo_fn(
          q_distr_out=q_distr_out_eval,
          smi_eta=smi_eta,
          batch=train_ds,
      )

      # Add two stages of ELBO to metrics dictionary
      metrics['elbo_stage_1'] = jnp.nanmean(elbo_dict_eval['stage_1'])
      metrics['elbo_stage_2'] = jnp.nanmean(elbo_dict_eval['stage_2'])

      # Log metrics to tensorboard
      metrics_to_log = ['elbo_stage_1', 'elbo_stage_2']
      for metric in metrics_to_log:
        summary_writer.scalar(
            tag=metric,
            value=metrics[metric],
            step=state.step,
        )

    if state.step % config.checkpoint_steps == 0:
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
      prng_key=next(prng_seq),
      flow_name=config.flow_name,
      flow_kwargs=config.flow_kwargs,
      num_samples_plot=config.num_samples_plot,
      smi_eta=smi_eta,
      summary_writer=summary_writer,
      workdir_png=workdir,
  )

  return state


# # For debugging
# config = get_config()
# eta = 0.001
# config.flow_kwargs.smi_eta = {'modules': [[1.0, eta]]}
# workdir = pathlib.Path.home() / f'smi/output/epidemiology/spline_eta_y_{eta:.3f}'
# train_and_evaluate(config, workdir)
