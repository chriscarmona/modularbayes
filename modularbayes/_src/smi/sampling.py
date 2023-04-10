"""Sampling from the Semi-Modular posterior using normalizing flows."""

import jax

import haiku as hk

from modularbayes._src.typing import (Any, Array, Callable, Dict, IntLike,
                                      Optional, PRNGKey, Tuple)


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
