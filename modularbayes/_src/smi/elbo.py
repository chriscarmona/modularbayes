"""Monte Carlo estimation of ELBO for Semi-Modular Inference."""
import jax
from jax import numpy as jnp

import haiku as hk

from modularbayes import sample_q
from modularbayes._src.typing import (Any, Array, Batch, Callable, Dict,
                                      PRNGKey, Tuple)


def elbo_smi(
    lambda_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    num_samples: int,
    batch: Batch,
    logprob_joint_fn: Callable,
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    flow_kwargs: Dict[str, Any],
    prior_hparams: Dict[str, float],
    model_params_tupleclass: type,
    model_params_cut_tupleclass: type,
    split_flow_fn_nocut: Callable,
    split_flow_fn_cut: Callable,
    smi_eta,
) -> Dict[str, Array]:
  """Estimate ELBO

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  # Sample from flow
  q_distr_out = sample_q(
      lambda_tuple=lambda_tuple,
      prng_key=prng_key,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      flow_kwargs=flow_kwargs,
      model_params_tupleclass=model_params_tupleclass,
      split_flow_fn_nocut=split_flow_fn_nocut,
      split_flow_fn_cut=split_flow_fn_cut,
      sample_shape=(num_samples,),
  )

  # ELBO stage 1: Power posterior
  log_prob_joint_stg1 = jax.vmap(lambda x: logprob_joint_fn(
      batch=batch,
      model_params=x,
      prior_hparams=prior_hparams,
      smi_eta=smi_eta,
  ))(
      q_distr_out['model_params_aux_sample'])
  log_q_stg1 = (q_distr_out['log_q_nocut'] + q_distr_out['log_q_cut_aux'])
  elbo_stg1 = log_prob_joint_stg1 - log_q_stg1

  # ELBO stage 2: Conventional posterior (with stop_gradient)
  model_params_stg2 = model_params_tupleclass(
      **{
          k: (v if k in
              model_params_cut_tupleclass._fields else jax.lax.stop_gradient(v))
          for k, v in q_distr_out['model_params_sample']._asdict().items()
      })
  log_prob_joint_stg2 = jax.vmap(lambda x: logprob_joint_fn(
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

  # Dictionary of ELBOs
  elbo_dict = {'stage_1': elbo_stg1, 'stage_2': elbo_stg2}

  return elbo_dict


def elbo_smi_vmpflow(
    lambda_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    num_samples: int,
    batch: Batch,
    logprob_joint_fn: Callable,
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    flow_kwargs: Dict[str, Any],
    prior_hparams: Dict[str, float],
    model_params_tupleclass: type,
    model_params_cut_tupleclass: type,
    split_flow_fn_nocut: Callable,
    split_flow_fn_cut: Callable,
    sample_eta_fn: Callable,
    sample_eta_kwargs: Dict[str, Any],
) -> Dict[str, Array]:
  """Estimate ELBO

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """
  prng_seq = hk.PRNGSequence(prng_key)

  # Sample eta values
  smi_etas = sample_eta_fn(
      prng_key=next(prng_seq),
      num_samples=num_samples,
      **sample_eta_kwargs,
  )

  # Sample from flow
  q_distr_out = sample_q(
      lambda_tuple=lambda_tuple,
      prng_key=prng_key,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      flow_kwargs=flow_kwargs,
      model_params_tupleclass=model_params_tupleclass,
      split_flow_fn_nocut=split_flow_fn_nocut,
      split_flow_fn_cut=split_flow_fn_cut,
      eta_values=(smi_etas[0] if len(smi_etas) == 1 else jnp.stack(
          smi_etas, axis=-1)),
  )

  # ELBO stage 1: Power posterior
  log_prob_joint_stg1 = jax.vmap(lambda x, y: logprob_joint_fn(
      batch=batch,
      model_params=x,
      prior_hparams=prior_hparams,
      smi_eta=y,
  ))(q_distr_out['model_params_aux_sample'], smi_etas)
  log_q_stg1 = (q_distr_out['log_q_nocut'] + q_distr_out['log_q_cut_aux'])
  elbo_stg1 = log_prob_joint_stg1 - log_q_stg1

  # ELBO stage 2: Conventional posterior (with stop_gradient)
  model_params_stg2 = model_params_tupleclass(
      **{
          k: (v if k in
              model_params_cut_tupleclass._fields else jax.lax.stop_gradient(v))
          for k, v in q_distr_out['model_params_sample']._asdict().items()
      })
  log_prob_joint_stg2 = jax.vmap(lambda x: logprob_joint_fn(
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

  # Dictionary of ELBOs
  elbo_dict = {'stage_1': elbo_stg1, 'stage_2': elbo_stg2}

  return elbo_dict


def elbo_smi_vmpmap(
    alpha_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    num_samples: int,
    batch: Batch,
    vmpmap_fn: hk.Transformed,
    lambda_init_tuple: Tuple[hk.Params],
    sample_eta_fn: Callable,
    sample_eta_kwargs: Dict[str, Any],
    elbo_smi_kwargs: Dict[str, Any],
) -> Dict[str, Array]:
  """Estimate ELBO.

  Monte Carlo estimate of ELBO for the two stages of variational SMI.
  Incorporates the stop_gradient operator for the secong stage.
  """

  prng_seq = hk.PRNGSequence(prng_key)

  # Sample eta values
  smi_etas = sample_eta_fn(
      prng_key=next(prng_seq),
      num_samples=num_samples,
      **sample_eta_kwargs,
  )

  lambda_tuple = [
      vmpmap_fn.apply(
          alpha_i,
          eta_values=(smi_etas[0] if len(smi_etas) == 1 else jnp.stack(
              smi_etas, axis=-1)),
          lambda_init=lambda_init_i)
      for alpha_i, lambda_init_i in zip(alpha_tuple, lambda_init_tuple)
  ]

  # Compute ELBO.
  elbo_dict = jax.vmap(lambda x, y, z: elbo_smi(
      lambda_tuple=x,
      prng_key=y,
      smi_eta=z,
      num_samples=1,  # Only one sample per eta value.
      batch=batch,
      **elbo_smi_kwargs,
  ))(lambda_tuple, jax.random.split(next(prng_seq), num_samples), smi_etas)

  return elbo_dict
