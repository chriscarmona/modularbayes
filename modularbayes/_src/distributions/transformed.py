"""Conditional Bijector applied to a Distribution."""

import distrax

from modularbayes._src.typing import Array, IntLike, PRNGKey, Sequence, Tuple, Union

import jax
import jax.numpy as jnp


class Transformed(distrax.Transformed):
  """Distribution transformed by a conditional bijective function.
  """

  def sample_and_log_prob_with_base(
      self,
      *,
      seed: Union[IntLike, PRNGKey],
      sample_shape: Union[IntLike, Sequence[IntLike]] = ()
  ) -> Tuple[Array]:

    x, lp_x = self.distribution.sample_and_log_prob(
        seed=seed, sample_shape=sample_shape)
    y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x)
    lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
    return y, lp_y, x
