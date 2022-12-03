"""Conditional Bijector abstract base class."""

import abc

from distrax._src.bijectors import bijector as base

from modularbayes._src.typing import Any, Array, Tuple


class ConditionalBijector(base.Bijector):
  """Differentiable bijection that knows to compute its Jacobian determinant.

  This class acts similarly to `distrax.Bijector`, but expand the functionality
  to take in an additional "context" argument that serves as an additional
  input to the bijector. Useful to define conditional random variables.
  """

  def forward(self, x: Array, context: Any) -> Array:
    """Computes y = f(x)."""
    y, _ = self.forward_and_log_det(x, context=context)
    return y

  def inverse(self, y: Array, context: Any) -> Array:
    """Computes x = f^{-1}(y)."""
    x, _ = self.inverse_and_log_det(y, context=context)
    return x

  def forward_log_det_jacobian(self, x: Array, context: Any) -> Array:
    """Computes log|det J(f)(x)|."""
    _, logdet = self.forward_and_log_det(x, context=context)
    return logdet

  def inverse_log_det_jacobian(self, y: Array, context: Any) -> Array:
    """Computes log|det J(f^{-1})(y)|."""
    _, logdet = self.inverse_and_log_det(y, context=context)
    return logdet

  @abc.abstractmethod
  def forward_and_log_det(
      self,
      x: Array,
      context: Any,
  ) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""

  def inverse_and_log_det(
      self,
      y: Array,
      context: Any,
  ) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    raise NotImplementedError(
        f"Bijector {self.name} does not implement `inverse_and_log_det`.")
