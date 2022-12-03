"""Conditional Chain Bijector for composing a sequence of Bijectors."""

import distrax

import modularbayes
from modularbayes._src.typing import Array, Optional, Tuple


class ConditionalChain(modularbayes.ConditionalBijector, distrax.Chain):
  """Composition of a sequence of bijectors into a single bijector.

  This class acts similarly to `distrax.Chain`, but expand the functionality to
  allow for conditional bijectors, i.e. bijectors that take in an additional
  context argument.
  """

  def forward(self, x: Array, context: Optional[Array] = None) -> Array:
    """Computes y = f(x)."""
    for bijector in reversed(self._bijectors):
      if isinstance(bijector, modularbayes.ConditionalBijector):
        x = bijector.forward(x, context)
      else:
        x = bijector.forward(x)
    return x

  def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
    """Computes x = f^{-1}(y)."""
    for bijector in self._bijectors:
      if isinstance(bijector, modularbayes.ConditionalBijector):
        y = bijector.inverse(y, context)
      else:
        y = bijector.inverse(y)
    return y

  def forward_and_log_det(
      self,
      x: Array,
      context: Optional[Array] = None,
  ) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""

    bijector = self._bijectors[-1]
    if isinstance(bijector, modularbayes.ConditionalBijector):
      x, log_det = bijector.forward_and_log_det(x, context)
    else:
      x, log_det = bijector.forward_and_log_det(x)

    for bijector in reversed(self._bijectors[:-1]):
      if isinstance(bijector, modularbayes.ConditionalBijector):
        x, ld = bijector.forward_and_log_det(x, context)
      else:
        x, ld = bijector.forward_and_log_det(x)
      log_det += ld
    return x, log_det

  def inverse_and_log_det(
      self,
      y: Array,
      context: Optional[Array] = None,
  ) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""

    bijector = self._bijectors[0]
    if isinstance(bijector, modularbayes.ConditionalBijector):
      y, log_det = bijector.inverse_and_log_det(y, context)
    else:
      y, log_det = bijector.inverse_and_log_det(y)

    for bijector in self._bijectors[1:]:
      if isinstance(bijector, modularbayes.ConditionalBijector):
        y, ld = bijector.inverse_and_log_det(y, context)
      else:
        y, ld = bijector.inverse_and_log_det(y)
      log_det += ld

    return y, log_det
