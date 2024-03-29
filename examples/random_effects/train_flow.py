"""Training a Normalizing Flow."""

import pathlib

from absl import logging

import numpy as np

from arviz import InferenceData

import jax
from jax import numpy as jnp
import haiku as hk
import optax
import distrax
import orbax.checkpoint
from objax.jaxboard import SummaryWriter, Summary

from modularbayes._src.typing import (
    Any,
    Array,
    Callable,
    ConfigDict,
    Dict,
    Optional,
    PRNGKey,
    SmiPosteriorStates,
    TrainState,
    Tuple,
)
from modularbayes import (
    elbo_smi,
    sample_q_nocut,
    sample_q_cutgivennocut,
    sample_q,
    train_step,
)

import flows
from flows import split_flow_nocut, split_flow_cut
from log_prob_fun import ModelParams, ModelParamsCut, SmiEta, logprob_joint
import plot

# Set high precision for matrix multiplication in jax
jax.config.update("jax_default_matmul_precision", "float32")

np.set_printoptions(suppress=True, precision=4)


def load_data(
    prng_key: PRNGKey,
    num_obs_groups: Array,
    loc_groups: Array,
    scale_groups: Array,
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

  dataset_dict = {
      "Y": loc_pointwise + z * scale_pointwise,
      "group": jnp.repeat(jnp.arange(num_groups), num_obs_groups),
      "num_obs_groups": num_obs_groups,
  }

  return dataset_dict


def init_state_tuple(
    prng_key: PRNGKey,
    config: ConfigDict,
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    sample_shape: Tuple[int],
    eta_values: Optional[Array] = None,
) -> SmiPosteriorStates:
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
  )["sample_base"]

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
    eta_values: Optional[Array] = None,
) -> None:
  """Print a summary of the trainable parameters."""
  prng_seq = hk.PRNGSequence(prng_key)

  logging.info("\nFlow no-cut parameters:\n")
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
  exec_str_ = tabulate_fn_(state_tuple[0], next(prng_seq))
  for line in exec_str_.split("\n"):
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
  )["sample_base"]

  logging.info("\nFlow cut parameters:\n")
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
  exec_str_ = tabulate_fn_(state_tuple[1], next(prng_seq))
  for line in exec_str_.split("\n"):
    logging.info(line)


def make_optimizer(
    lr_schedule_name,
    lr_schedule_kwargs,
    grad_clip_value,
) -> optax.GradientTransformation:
  """Define optimizer."""
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
  loss_avg = -(jnp.nanmean(elbo_dict["stage_1"] + elbo_dict["stage_2"]))

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
  """Sample Model parameters in Arviz format."""
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
                                 q_distr_out["model_params_sample"])
  # Create InferenceData object
  az_data = plot.arviz_from_samples(
      dataset=dataset,
      model_params=model_params_az,
  )
  return az_data


def elpd_waic(
    model_params_sample: ModelParams,
    dataset: Dict[str, Any],
) -> float:
  """Compute ELPD WAIC.

  Estimates the ELPD based on a Monte Carlo approximations. Using WAIC.
  """
  num_samples, _ = model_params_sample.beta.shape

  # Posterior predictive distribution
  predictive_dist = distrax.Normal(
      loc=model_params_sample.beta[:, dataset["group"]],
      scale=model_params_sample.sigma[:, dataset["group"]],
  )

  # Compute LPD
  loglik_pointwise_insample = predictive_dist.log_prob(dataset["Y"])
  lpd_pointwise = jax.scipy.special.logsumexp(
      loglik_pointwise_insample, axis=0) - jnp.log(num_samples)
  lpd = lpd_pointwise.mean(axis=-1)

  # Estimated effective number of parameters
  # Variance over samples
  p_waic_pointwise = jnp.var(loglik_pointwise_insample, axis=0)
  p_waic = p_waic_pointwise.mean(axis=-1)

  # ELPD WAIC
  elpd_out = lpd - p_waic

  return elpd_out


def elpd_truth_mc(
    model_params_sample: ModelParams,
    y_new: Array,
) -> float:
  """Compute ELPD.

  Estimates the ELPD based on a Monte Carlo approximation. Assuming we have a
  sample of new data from the true generative model. Compute the
  Expected Log-Predictive density over the new data.
  """

  num_samples, _ = model_params_sample.beta.shape

  # Posterior predictive distribution
  predictive_dist = distrax.Normal(
      loc=model_params_sample.beta,
      scale=model_params_sample.sigma,
  )

  loglik_pointwise_outofsample = predictive_dist.log_prob(y_new)

  # Computed log-pointwise predictive density
  # Average over posterior samples
  elpd_mc_pointwise = jax.scipy.special.logsumexp(
      loglik_pointwise_outofsample, axis=0) - jnp.log(
          num_samples)  # colLogMeanExps
  # Average over "observations" (samples from true generative model)
  elpd_out = elpd_mc_pointwise.mean(axis=-1)

  return elpd_out


def train_and_evaluate(config: ConfigDict, workdir: str) -> TrainState:
  """Model training and evaluation.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  # Add trailing slash
  workdir = workdir.rstrip("/") + "/"
  pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)

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

  # Set eta for modules
  smi_eta = SmiEta(groups=jnp.array(config.smi_eta_groups))

  # num_groups is also a parameter of the flow,
  # as it define its dimension
  config.flow_kwargs.num_groups = config.num_groups
  # Also is_smi modifies the dimension of the flow, due to the duplicated params
  config.flow_kwargs.is_smi = smi_eta is not None

  # Functions that generate the NF for no-cut params
  flow_get_fn_nocut = getattr(flows, "get_q_nocut_" + config.flow_name)
  # Functions that generate the NF for cut params (conditional on no-cut params)
  flow_get_fn_cutgivennocut = getattr(flows,
                                      "get_q_cutgivennocut_" + config.flow_name)

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
          directory=str(pathlib.Path(workdir) / "checkpoints" / state_name),
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
            "num_samples": config.num_samples_elbo,
            "logprob_joint_fn": logprob_joint,
            "flow_get_fn_nocut": flow_get_fn_nocut,
            "flow_get_fn_cutgivennocut": flow_get_fn_cutgivennocut,
            "flow_kwargs": config.flow_kwargs,
            "prior_hparams": config.prior_hparams,
            "model_params_tupleclass": ModelParams,
            "model_params_cut_tupleclass": ModelParamsCut,
            "split_flow_fn_nocut": split_flow_nocut,
            "split_flow_fn_cut": split_flow_cut,
            "smi_eta": smi_eta,
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

  if int(state_tuple[0].step) < config.training_steps:
    logging.info("Training variational posterior...")

  # Reset random key sequence
  prng_seq = hk.PRNGSequence(config.seed)

  tensorboard = SummaryWriter(workdir)

  while int(state_tuple[0].step) < config.training_steps:
    summary = Summary()

    # SGD step
    state_tuple_, metrics = train_step_jit(
        state_tuple=state_tuple,
        batch=train_ds,
        prng_key=next(prng_seq),
    )
    if jax.lax.is_finite(metrics["train_loss"]):
      state_tuple = state_tuple_

    summary.scalar(tag="train_loss", value=metrics["train_loss"])

    if int(state_tuple[0].step) == 1:
      logging.info(
          "STEP: %5d; training loss: %.3f",
          int(state_tuple[0].step),
          metrics["train_loss"],
      )

    # Log learning rate
    summary.scalar(
        tag="learning_rate",
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(int(state_tuple[0].step)),
    )

    # Metrics for evaluation
    if int(state_tuple[0].step) % config.eval_steps == 0:
      logging.info(
          "STEP: %5d; training loss: %.3f",
          int(state_tuple[0].step),
          metrics["train_loss"],
      )

      elbo_dict = elbo_validation_jit(
          state_tuple=state_tuple,
          batch=train_ds,
          prng_key=next(prng_seq),
          smi_eta=smi_eta,
      )
      for k, v in elbo_dict.items():
        summary.scalar(tag=f"elbo_{k}", value=v.mean())

    # Plots to monitor during training
    if (config.log_img_steps
        is not None) and (int(state_tuple[0].step) % config.log_img_steps == 0):
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
          show_sigma_trace=False,
          show_beta_trace=False,
          show_tau_trace=False,
          betasigma_pairplot_groups=(0, 1, 2),
          tausigma_pairplot_groups=(0, 1, 2),
          suffix=config.plot_suffix,
          workdir_png=workdir,
          summary=summary,
      )
      logging.info("\t\t...done logging plots.")

    # Write summary to tensorboard
    tensorboard.write(summary=summary, step=int(state_tuple[0].step))

    # Save checkpoints
    for state, mngr in zip(state_tuple, orbax_ckpt_mngrs):
      mngr.save(step=int(state.step), items=state)

    # Wait until computations are done before the next step
    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  logging.info("Final training step: %i", int(state_tuple[0].step))

  # Last plot of posteriors
  summary = Summary()
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
      show_sigma_trace=False,
      show_beta_trace=False,
      show_tau_trace=False,
      betasigma_pairplot_groups=(0, 1, 2),
      tausigma_pairplot_groups=(0, 1, 2),
      suffix=config.plot_suffix,
      workdir_png=workdir,
      summary=summary,
  )
  # Write metrics to tensorboard
  tensorboard.write(summary=summary, step=state_tuple[0].step)
  # Close tensorboard writer
  tensorboard.close()

  return state_tuple
