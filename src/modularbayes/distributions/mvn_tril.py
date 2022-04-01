"""Multivariate Normal distribution classes."""

from jax import numpy as jnp

import distrax
from distrax._src.bijectors import bijector as bjct_base
from distrax._src.distributions import distribution as dist_base

from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

Array = jnp.ndarray


class MultivariateNormalTriL(distrax.Transformed):
  """Multivariate normal distribution."""
  equiv_tfp_cls = tfd.MultivariateNormalTriL

  def __init__(self, loc: Array, scale_tril: Array):
    """Multivariate normal distribution.

    Args:
      loc: mean vector. If this is set to `None`, `loc` is
        implicitly `0`.
      scale_tril: Floating-point, lower-triangular `Tensor` with non-zero
        diagonal elements.
    """

    flow_layers = []

    # Layer 1: Scale by cov_chol, a lower triangular matrix
    flow_layers.append(tfb.ScaleMatvecTriL(scale_tril=scale_tril))

    # Layer 2: Shift by loc
    flow_layers.append(distrax.Block(tfb.Shift(shift=loc), 1))

    # Chain all layers together
    flow = distrax.Chain(flow_layers[::-1])

    base_distribution = distrax.MultivariateNormalDiag(
        loc=jnp.zeros_like(loc), scale_diag=jnp.ones_like(loc))

    super().__init__(base_distribution, flow)
