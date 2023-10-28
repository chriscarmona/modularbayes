"""Training a Variational Meta-Posterior, using the VMP flow."""

import pathlib

from absl import logging

import numpy as np

import jax
from jax import numpy as jnp
import haiku as hk
import optax
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
    TrainState,
    Tuple,
)
from modularbayes import elbo_smi_vmpflow, train_step

from train_flow import (
    init_state_tuple,
    load_data,
    print_trainable_params,
    sample_q_as_az,
)
import flows
from flows import split_flow_nocut, split_flow_cut
from log_prob_fun import (
    ModelParams,
    ModelParamsCut,
    SmiEta,
    logprob_joint,
    sample_eta_values,
)
import plot

# Set high precision for matrix multiplication in jax
jax.config.update("jax_default_matmul_precision", "float32")

np.set_printoptions(suppress=True, precision=4)


def loss_fn(params_tuple: Tuple[hk.Params], *args, **kwargs) -> Array:
  """Define training loss function."""

  ### Compute ELBO ###
  elbo_dict = elbo_smi_vmpflow(lambda_tuple=params_tuple, *args, **kwargs)

  # Our loss is the Negative ELBO
  loss_avg = -(jnp.nanmean(elbo_dict["stage_1"] + elbo_dict["stage_2"]))

  return loss_avg


def log_images(
    state_tuple: Tuple[TrainState],
    prng_key: PRNGKey,
    config: ConfigDict,
    dataset: Dict[str, Any],
    num_samples_plot: int,
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    summary: Optional[Summary] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plots to monitor during training."""

  prng_seq = hk.PRNGSequence(prng_key)

  assert len(config.smi_eta_cancer_plot) > 0, "No eta values to plot"
  assert all((x >= 0.0) and (x <= 1.0)
             for x in config.smi_eta_cancer_plot), "Invalid eta values"

  # Plot posterior samples
  for eta_cancer_i in config.smi_eta_cancer_plot:
    # Define eta with a single value
    smi_etas = SmiEta(
        hpv=jnp.ones((num_samples_plot, 1)),
        cancer=eta_cancer_i * jnp.ones((num_samples_plot, 1)),
    )
    # Sample from flow
    az_data = sample_q_as_az(
        lambda_tuple=tuple(x.params for x in state_tuple),
        dataset=dataset,
        prng_key=next(prng_seq),
        flow_get_fn_nocut=flow_get_fn_nocut,
        flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
        flow_kwargs=config.flow_kwargs,
        sample_shape=(num_samples_plot,),
        eta_values=(smi_etas[0] if len(smi_etas) == 1 else jnp.concatenate(
            smi_etas, axis=-1)),
    )
    plot.posterior_plots(
        az_data=az_data,
        show_phi_trace=False,
        show_theta_trace=False,
        show_loglinear_scatter=True,
        show_theta_pairplot=True,
        eta=eta_cancer_i,
        suffix=f"_eta_cancer_{float(eta_cancer_i):.3f}",
        workdir_png=workdir_png,
        summary=summary,
    )


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

  # Full dataset used everytime
  # Small data, no need to batch
  train_ds = load_data()

  phi_dim = train_ds["Z"].shape[0]
  theta_dim = 2

  # phi_dim and theta_dim are also arguments of the flow,
  # as they define its dimension
  config.flow_kwargs.phi_dim = phi_dim
  config.flow_kwargs.theta_dim = theta_dim
  # Also is_smi modifies the dimension of the flow, due to the duplicated params
  config.flow_kwargs.is_smi = True

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
      eta_values=jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
  )

  # Print a summary of the networks architecture
  print_trainable_params(
      state_tuple=state_tuple,
      prng_key=next(prng_seq),
      config=config,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      sample_shape=(config.num_samples_elbo,),
      eta_values=jnp.ones((config.num_samples_elbo, config.smi_eta_dim)),
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
            "sample_eta_fn": sample_eta_values,
            "sample_eta_kwargs": {
                "eta_sampling_a": config.eta_sampling_a,
                "eta_sampling_b": config.eta_sampling_b,
            },
        },
    )

  @jax.jit
  def elbo_validation_jit(state_tuple, batch, prng_key):
    return elbo_smi_vmpflow(
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
        sample_eta_fn=sample_eta_values,
        sample_eta_kwargs={
            "eta_sampling_a": 1.0,
            "eta_sampling_b": 1.0
        },
    )

  if int(state_tuple[0].step) < config.training_steps:
    logging.info("Training Variational Meta-Posterior (VMP-flow)...")

  # Reset random key sequence
  prng_seq = hk.PRNGSequence(config.seed)

  tensorboard = SummaryWriter(workdir)

  while int(state_tuple[0].step) < config.training_steps:
    summary = Summary()

    # Plots to monitor training
    if (config.log_img_steps
        is not None) and (int(state_tuple[0].step) % config.log_img_steps == 0):
      logging.info("\t\t Logging plots...")
      log_images(
          state_tuple=state_tuple,
          prng_key=next(prng_seq),
          config=config,
          dataset=train_ds,
          num_samples_plot=config.num_samples_plot,
          flow_get_fn_nocut=flow_get_fn_nocut,
          flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
          summary=summary,
          workdir_png=workdir,
      )
      logging.info("\t\t...done logging plots.")

    # Log learning rate
    summary.scalar(
        tag="learning_rate",
        value=getattr(optax, config.optim_kwargs.lr_schedule_name)(
            **config.optim_kwargs.lr_schedule_kwargs)(int(state_tuple[0].step)),
    )

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
      )
      for k, v in elbo_dict.items():
        summary.scalar(tag=f"elbo_{k}", value=v.mean())

    # Write metrics to tensorboard
    tensorboard.write(summary=summary, step=int(state_tuple[0].step))

    for state, mngr in zip(state_tuple, orbax_ckpt_mngrs):
      mngr.save(step=int(state.step), items=state)

  logging.info("Final training step: %i", int(state_tuple[0].step))

  # Last plot of posteriors
  log_images(
      state_tuple=state_tuple,
      prng_key=next(prng_seq),
      config=config,
      dataset=train_ds,
      num_samples_plot=config.num_samples_plot,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      summary=summary,
      workdir_png=workdir,
  )
  # Write metrics to tensorboard
  tensorboard.write(summary=summary, step=int(state_tuple[0].step))
  # Close tensorboard writer
  tensorboard.close()

  return state_tuple
