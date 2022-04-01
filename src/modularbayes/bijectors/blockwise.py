"""Blockwise bijector.

Reimplementation from tensorflow probability.
"""

from typing import List, Sequence, Tuple
import itertools
from jax import numpy as jnp

from distrax._src.bijectors import bijector as base
from distrax._src.utils import conversion

Array = base.Array
BijectorLike = base.BijectorLike
BijectorT = base.BijectorT


class Blockwise(base.Bijector):
  """Bijector which applies a list of bijectors to blocks of a `Tensor`.

  More specifically, given [F_0, F_1, ... F_n] which are scalar or vector
  bijectors this bijector creates a transformation which operates on the vector
  [x_0, ... x_n] with the transformation [F_0(x_0), F_1(x_1) ..., F_n(x_n)]
  where x_0, ..., x_n are blocks (partitions) of the vector.

  Example Use:

  ```python
  blockwise = tfb.Blockwise(
      bijectors=[tfb.Exp(), tfb.Sigmoid()], block_sizes=[2, 1]
    )
  y = blockwise.forward(x)

  # Equivalent to:
  x_0, x_1 = tf.split(x, [2, 1], axis=-1)
  y_0 = tfb.Exp().forward(x_0)
  y_1 = tfb.Sigmoid().forward(x_1)
  y = tf.concat([y_0, y_1], axis=-1)
  ```

  Keyword arguments can be passed to the inner bijectors by utilizing the inner
  bijector names, e.g.:

  ```python
  blockwise = tfb.Blockwise([Bijector1(name='b1'), Bijector2(name='b2')])
  y = blockwise.forward(x, b1={'arg': 1}, b2={'arg': 2})

  # Equivalent to:
  x_0, x_1 = tf.split(x, [1, 1], axis=-1)
  y_0 = Bijector1().forward(x_0, arg=1)
  y_1 = Bijector2().forward(x_1, arg=2)
  y = tf.concat([y_0, y_1], axis=-1)
  ```

  """

  def __init__(
      self,
      bijectors: Sequence[BijectorLike],
      block_sizes=None,
  ):
    """Creates the bijector.

    Args:
      bijectors: A non-empty sequence of bijectors.
      block_sizes: A 1-D integer `Tensor` with each element signifying the
        length of the block of the input vector to pass to the corresponding
        bijector. The length of `block_sizes` must be be equal to the length of
        `bijectors`. If left as None, a vector of 1's is used.
    """

    if not bijectors:
      raise ValueError("The sequence of bijectors cannot be empty.")
    self._bijectors = [conversion.as_bijector(b) for b in bijectors]

    assert all(b > 0 for b in block_sizes)
    self._block_sizes = [int(b) for b in block_sizes]
    self._block_sizes_split = list(itertools.accumulate(block_sizes))[:-1]

    #TODO: Enable bijectors other than 1D
    assert all(bijector.event_ndims_in == 1 for bijector in self._bijectors)
    event_ndims_in = 1
    event_ndims_out = 1

    is_constant_jacobian = all(b.is_constant_jacobian for b in self._bijectors)
    is_constant_log_det = all(b.is_constant_log_det for b in self._bijectors)

    super().__init__(
        event_ndims_in=event_ndims_in,
        event_ndims_out=event_ndims_out,
        is_constant_jacobian=is_constant_jacobian,
        is_constant_log_det=is_constant_log_det)

  @property
  def bijectors(self) -> List[BijectorT]:
    return self._bijectors

  @property
  def block_sizes(self) -> List[int]:
    return self._block_sizes

  def forward(self, x: Array) -> Array:
    """Computes y = f(x)."""
    x_split = jnp.split(x, self._block_sizes_split, axis=-1)
    y_split = [
        bijector_i.forward(x_i)
        for bijector_i, x_i in zip(self._bijectors, x_split)
    ]
    y = jnp.concatenate(y_split, axis=-1)
    return y

  def inverse(self, y: Array) -> Array:
    """Computes x = f^{-1}(y)."""
    y_split = jnp.split(y, self._block_sizes_split, axis=-1)
    x_split = [
        bijector_i.inverse(x_i)
        for bijector_i, x_i in zip(self._bijectors, y_split)
    ]
    x = jnp.concatenate(x_split, axis=-1)
    return x

  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""

    x_split = jnp.split(x, self._block_sizes_split, axis=-1)
    y_and_log_det_split = [
        bijector_i.forward_and_log_det(x_i)
        for bijector_i, x_i in zip(self._bijectors, x_split)
    ]
    y_split, log_det_split = list(zip(*y_and_log_det_split))
    y = jnp.concatenate(y_split, axis=-1)
    log_det = sum(log_det_split)
    return y, log_det

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    y_split = jnp.split(y, self._block_sizes_split, axis=-1)
    x_and_log_det_split = [
        bijector_i.inverse_and_log_det(x_i)
        for bijector_i, x_i in zip(self._bijectors, y_split)
    ]
    x_split, log_det_split = list(zip(*x_and_log_det_split))
    x = jnp.concatenate(x_split, axis=-1)
    log_det = sum(log_det_split)
    return x, log_det
