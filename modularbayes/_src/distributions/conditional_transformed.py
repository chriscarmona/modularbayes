"""Conditional Bijector applied to a Distribution."""

import functools
import operator

import jax
import jax.numpy as jnp

import distrax
from distrax._src.distributions import distribution as dist_base

from modularbayes._src.typing import (Array, Tuple, Union, IntLike, PRNGKey,
                                 Sequence)


class ConditionalTransformed(distrax.Transformed):
  """Distribution transformed by a conditional bijective function."""

  def log_prob(self, value: Array, context: Array) -> Array:
    """See `Distribution.log_prob`."""
    x, ildj_y = self.bijector.inverse_and_log_det(y=value, context=context)
    lp_x = self.distribution.log_prob(x)
    lp_y = lp_x + ildj_y
    return lp_y

  def _sample_n(self, key: PRNGKey, n: int, context: Array) -> Array:
    """Returns `n` samples."""
    x = self.distribution.sample(seed=key, sample_shape=n)
    y = jax.vmap(self.bijector.forward)(x, context)
    return y

  def _sample_n_and_log_prob(
      self,
      key: PRNGKey,
      n: int,
      context: Array,
  ) -> Tuple[Array, Array]:
    """Returns `n` samples and their log probs.

    This function is more efficient than calling `sample` and `log_prob`
    separately, because it uses only the forward methods of the bijector. It
    also works for bijectors that don't implement inverse methods.

    Args:
      key: PRNG key.
      n: Number of samples to generate.

    Returns:
      A tuple of `n` samples and their log probs.
    """
    x, lp_x = self.distribution.sample_and_log_prob(seed=key, sample_shape=n)
    y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x, context)
    lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
    return y, lp_y

  def sample_and_log_prob(
      self,
      *,
      seed: Union[IntLike, PRNGKey],
      sample_shape: Union[IntLike, Sequence[IntLike]] = (),
      context: Array,
  ) -> Tuple[Array, Array]:
    """Returns a sample and associated log probability. See `sample`."""
    rng, sample_shape = dist_base.convert_seed_and_sample_shape(
        seed, sample_shape)
    num_samples = functools.reduce(operator.mul, sample_shape, 1)  # product

    samples, log_prob = self._sample_n_and_log_prob(rng, num_samples, context)

    samples = samples.reshape(sample_shape + samples.shape[1:])
    log_prob = log_prob.reshape(sample_shape + log_prob.shape[1:])
    return samples, log_prob

  def sample_and_log_prob_with_base(
      self,
      *,
      seed: Union[IntLike, PRNGKey],
      sample_shape: Union[IntLike, Sequence[IntLike]] = (),
      context: Array,
  ) -> Tuple[Array]:
    """Returns a sample and associated log probability. See `sample`."""
    x, lp_x = self.distribution.sample_and_log_prob(
        seed=seed, sample_shape=sample_shape)
    y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x, context)
    lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
    return y, lp_y, x
