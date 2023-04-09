import jax
from jax import numpy as jnp

import haiku as hk

from modularbayes._src.typing import (Any, Array, Batch, Callable, Dict,
                                      IntLike, Optional, PRNGKey, Tuple)


@hk.transform
def sample_q_nocut(
    flow_get_fn: Callable,
    flow_kwargs: Dict[str, Any],
    split_flow_fn: Callable,
    sample_shape: Optional[Tuple[IntLike]] = None,
    eta_values: Optional[Array] = None,
) -> Dict[str, Any]:
  """Sample from variational posterior for no-cut parameters."""

  q_output = {}

  # Define normalizing flows
  q_distr = flow_get_fn(**flow_kwargs)

  assert sample_shape is not None or eta_values is not None, (
      'Either sample_shape or eta_values must be provided.')
  assert sample_shape is None or eta_values is None, (
      'Only one of sample_shape or eta_values must be provided.')

  if eta_values is None:
    kwargs_distr_ = {'sample_shape': sample_shape}
  else:
    kwargs_distr_ = {
        'sample_shape': (eta_values.shape[0],),
        'context': (eta_values, None)
    }

  # Sample from flows
  (sample_flow_concat, sample_logprob,
   sample_base) = q_distr.sample_and_log_prob_with_base(
       seed=hk.next_rng_key(), **kwargs_distr_)

  # Split flow into model parameters
  q_output['sample'] = jax.vmap(lambda x: split_flow_fn(
      concat_params=x,
      **flow_kwargs,
  ))(
      sample_flow_concat)

  # sample from base distribution
  q_output['sample_base'] = sample_base

  # variational posterior evaluated in the sample
  q_output['sample_logprob'] = sample_logprob

  return q_output


@hk.transform
def sample_q_cutgivennocut(
    flow_get_fn: Callable,
    flow_kwargs: Dict[str, Any],
    nocut_base_sample: Array,
    split_flow_fn: Callable,
    eta_values: Optional[Array] = None,
) -> Dict[str, Any]:
  """Sample from variational posterior for cut parameters
  Conditional on values of no-cut parameters."""

  q_output = {}

  if eta_values is None:
    kwargs_distr_ = {
        'sample_shape': (nocut_base_sample.shape[0],),
        'context': nocut_base_sample,
    }
  else:
    assert nocut_base_sample.shape[0] == eta_values.shape[0], (
        'First diension of nocut_base_sample and eta_valuesmust match.')
    kwargs_distr_ = {
        'sample_shape': (nocut_base_sample.shape[0],),
        'context': (eta_values, nocut_base_sample),
    }

  # Define normalizing flows
  q_distr = flow_get_fn(**flow_kwargs)

  # Sample from flows
  (sample, sample_logprob) = q_distr.sample_and_log_prob(
      seed=hk.next_rng_key(), **kwargs_distr_)

  # Split flow into model parameters
  q_output['sample'] = jax.vmap(lambda x: split_flow_fn(
      concat_params=x,
      **flow_kwargs,
  ))(
      sample)

  # variational posterior evaluated in the sample
  q_output['sample_logprob'] = sample_logprob

  return q_output


def sample_q(
    lambda_tuple: Tuple[hk.Params],
    prng_key: PRNGKey,
    flow_get_fn_nocut: Callable,
    flow_get_fn_cutgivennocut: Callable,
    flow_kwargs: Dict[str, Any],
    model_params_tupleclass: type,
    split_flow_fn_nocut: Callable,
    split_flow_fn_cut: Callable,
    sample_shape: Optional[Tuple[IntLike]] = None,
    eta_values: Optional[Array] = None,
) -> Dict[str, Any]:
  """Sample from model posterior"""

  prng_seq = hk.PRNGSequence(prng_key)

  assert sample_shape is not None or eta_values is not None, (
      'Either sample_shape or eta_values must be provided.')
  assert sample_shape is None or eta_values is None, (
      'Only one of sample_shape or eta_values must be provided.')

  q_output = {}

  # Sample from q(no_cut_params)
  q_output_nocut_ = sample_q_nocut.apply(
      lambda_tuple[0],
      next(prng_seq),
      flow_get_fn=flow_get_fn_nocut,
      flow_kwargs=flow_kwargs,
      split_flow_fn=split_flow_fn_nocut,
      sample_shape=sample_shape,
      eta_values=eta_values,
  )

  # Sample from q(cut_params|no_cut_params)
  q_output_cut_ = sample_q_cutgivennocut.apply(
      lambda_tuple[1],
      next(prng_seq),
      flow_get_fn=flow_get_fn_cutgivennocut,
      flow_kwargs=flow_kwargs,
      split_flow_fn=split_flow_fn_cut,
      nocut_base_sample=q_output_nocut_['sample_base'],
      eta_values=eta_values,
  )

  q_output['model_params_sample'] = model_params_tupleclass(**{
      **q_output_nocut_['sample']._asdict(),
      **q_output_cut_['sample']._asdict(),
  })
  q_output['log_q_nocut'] = q_output_nocut_['sample_logprob']
  q_output['log_q_cut'] = q_output_cut_['sample_logprob']

  if flow_kwargs.is_smi:
    q_output_cut_aux_ = sample_q_cutgivennocut.apply(
        lambda_tuple[2],
        next(prng_seq),
        flow_get_fn=flow_get_fn_cutgivennocut,
        flow_kwargs=flow_kwargs,
        split_flow_fn=split_flow_fn_cut,
        nocut_base_sample=q_output_nocut_['sample_base'],
        eta_values=eta_values,
    )
    q_output['model_params_aux_sample'] = model_params_tupleclass(
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
      lambda_tuple=params_tuple,
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


def elbo_estimate_meta(
    params_tuple: Tuple[hk.Params],
    batch: Batch,
    prng_key: PRNGKey,
    num_samples: int,
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
      lambda_tuple=params_tuple,
      prng_key=prng_key,
      flow_get_fn_nocut=flow_get_fn_nocut,
      flow_get_fn_cutgivennocut=flow_get_fn_cutgivennocut,
      flow_kwargs=flow_kwargs,
      model_params_tupleclass=model_params_tupleclass,
      split_flow_fn_nocut=split_flow_fn_nocut,
      split_flow_fn_cut=split_flow_fn_cut,
      eta_values=jnp.stack(smi_etas, axis=-1),
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
