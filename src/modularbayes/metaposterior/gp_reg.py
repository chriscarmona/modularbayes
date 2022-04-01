"""Define GP map for Variational Meta-Posteriors"""

import math

import haiku as hk
import jax
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp

from modularbayes.typing import Any, Array, Kernel, Mapping, List, Optional

kernels = tfp.math.psd_kernels


def gp_reg_mean(
    x_new: Array,
    x: Array,
    y: Array,
    kernel: Kernel,
    y_sigma: float = 0.,
    k_x_inv: Optional[Array] = None,
) -> Array:
  """Predictive Mean using GP regression.

  Predict new observations using a naive fit of a Gaussian Process regresssion.
  E(f(x_new)|x,y) = K(x_new,x) [K(x,x)+sigma^2 * I] y

  f ~ GP(K)
  y ~ N( f(x), y_sigma)

  Args:
    x_new: Array with the new features (x) to be predicted.
    x: Array with the observed features values.
    y: Array with the observed responses.
    kernel: Kernel defining the GP.
    y_sigma: positive value defining the variance of the data.
    k_x_inv: (Optional) Matrix with the inverse of the kernel evaluate in the
      data for the observed features values.
  """
  k_xnew_x = kernel.matrix(x_new, x)
  if k_x_inv is None:
    k_x = kernel.matrix(x, x) + y_sigma**2 * jnp.eye(x.shape[0])
    k_x_inv = jnp.linalg.inv(k_x)
  aux1 = jnp.einsum('mn,no->mo', k_x_inv, y, precision='highest')
  f_new_mean = jnp.einsum('rn,no->ro', k_xnew_x, aux1, precision='highest')
  return f_new_mean


class VmpGP(hk.Module):
  """Trainable mapping from eta to normalizing flow parameters."""

  def __init__(
      self,
      params_flow_init: hk.Params,
      eta_knots: List[float],
      kernel_name: str,
      kernel_kwargs: Mapping[str, Any],
      y_sigma: float,
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
    self.num_eta_knots, _ = self.eta_knots.shape

    self.kernel_name = kernel_name
    self.kernel_kwargs = kernel_kwargs
    self.y_sigma = y_sigma

    cov_eta_knots = getattr(kernels,
                            self.kernel_name)(**self.kernel_kwargs).matrix(
                                self.eta_knots, self.eta_knots)
    cov_eta_knots = cov_eta_knots + self.y_sigma * jnp.eye(
        self.eta_knots.shape[0])
    self.cov_eta_knots_inv = jnp.linalg.inv(cov_eta_knots)

  def __call__(self, eta: Array) -> List[hk.Params]:
    """Trainable mapping from eta to normalizing flow parameters."""
    assert eta.ndim == 2
    assert eta.shape[-1] == self.eta_knots.shape[-1]

    num_eta_new, _ = eta.shape

    f_eta_knots = hk.get_parameter(
        "f_eta_knots",
        (self.num_eta_knots, self.output_dim),
        init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal'),
    )

    f_eta_knots_mean = jnp.mean(f_eta_knots, axis=0, keepdims=True)
    f_eta_knots_sd = jnp.std(f_eta_knots, axis=0, keepdims=True)
    # Avoid zero division
    f_eta_knots_sd = jnp.where(f_eta_knots_sd < 1e-6, 1., f_eta_knots_sd)

    flow_params_merged = gp_reg_mean(
        x_new=eta,
        x=self.eta_knots,
        y=(f_eta_knots - f_eta_knots_mean) / f_eta_knots_sd,
        kernel=getattr(kernels, self.kernel_name)(**self.kernel_kwargs),
        y_sigma=self.y_sigma,
        k_x_inv=self.cov_eta_knots_inv,
    )
    flow_params_merged = f_eta_knots_mean + flow_params_merged * f_eta_knots_sd

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
