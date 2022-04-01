"""Define Cubic Spline map for Variational Meta-Posteriors"""

import math

import jax
from jax import numpy as jnp
import haiku as hk

from modularbayes.typing import Array, List, Optional


def _d_k(x: Array, knots: Array, k: int):
  """Auxiliary function for Natural Cubic Splines.
  Equation 5.5 from ESL 2nd edition.
  """
  assert x.ndim == 2
  assert knots.ndim == 2
  assert (k > 0) & (k < knots.shape[0])
  knot_k = knots[k - 1]
  knot_last = knots[-1]
  a = x - knot_k
  b = x - knot_last
  d_k = (jnp.where(a > 0, a, 0)**3 - jnp.where(b > 0, b, 0)**3) / (
      knot_last - knot_k)

  return d_k


def natural_cubic_spline(
    x: Array,
    knots: Array,
    coef: Array,
):
  """Return values
  """
  assert x.ndim == 2
  assert knots.ndim == 2
  assert coef.ndim == 1

  num_knots = knots.shape[0]
  assert num_knots > 1
  assert coef.shape[0] == num_knots

  # Create basis functions
  basis_funs = []
  basis_funs.append(jnp.ones_like(x))
  basis_funs.append(x)
  basis_funs = basis_funs + [
      _d_k(x=x, knots=knots, k=k) - _d_k(x=x, knots=knots, k=num_knots - 1)
      for k in range(1, num_knots - 1)
  ]
  basis_funs = jnp.concatenate(basis_funs, axis=-1)
  f_x = jnp.matmul(basis_funs, coef.reshape(-1, 1), precision='highest')

  return f_x


class VmpCubicSpline(hk.Module):
  """Trainable mapping from eta to normalizing flow parameters."""

  def __init__(
      self,
      params_flow_init: hk.Params,
      eta_knots: List[float],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    # Variational parameters to be produced
    # Tree definition
    leaves, self.params_flow_treedef = jax.tree_util.tree_flatten(
        params_flow_init)
    self.params_flow_shapes = [x.shape for x in leaves]
    # Total number of output paramemters by the vmp
    self.output_dim = sum([x.flatten().shape[0] for x in leaves])

    self._eta_knots = eta_knots
    assert self.eta_knots.ndim == 2
    self.num_spline_coef, _ = self.eta_knots.shape

  def __call__(self, eta: Array) -> List[hk.Params]:

    assert eta.ndim == 2

    assert eta.shape[-1] == self.eta_knots.shape[-1]

    num_eta_new, _ = eta.shape

    # Initialize spline intercept on params_flow_init
    spline_coef_0 = hk.get_parameter(
        "spline_coef_0",
        (1, self.output_dim),
        init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'),
    )
    # Initialize other spline coefficients near zero
    spline_coef_1 = hk.get_parameter(
        "spline_coef_1",
        (self.num_spline_coef - 1, self.output_dim),
        init=hk.initializers.VarianceScaling(0.001, 'fan_in',
                                             'truncated_normal'),
    )
    # All spline coefficients concatenated
    spline_coef = jnp.concatenate([spline_coef_0, spline_coef_1], axis=0)

    flow_params_merged = jax.vmap(
        lambda x: natural_cubic_spline(
            x=eta,
            knots=self.eta_knots,
            coef=x,
        ),
        in_axes=1,
        out_axes=1)(spline_coef).squeeze(-1)

    leaves_eta = []
    for i in range(len(self.params_flow_shapes) - 1):
      param_i, flow_params_merged = jnp.split(
          flow_params_merged, (math.prod(self.params_flow_shapes[i]),), axis=-1)
      leaves_eta.append(
          param_i.reshape((num_eta_new,) + self.params_flow_shapes[i]))
    leaves_eta.append(
        flow_params_merged.reshape((num_eta_new,) +
                                   self.params_flow_shapes[-1]))

    params_flow_out = jax.tree_util.tree_unflatten(
        treedef=self.params_flow_treedef, leaves=leaves_eta)

    return params_flow_out

  @property
  def eta_knots(self):
    return jnp.array(self._eta_knots)
