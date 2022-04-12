"""Define normalizing flows for the Epidemiology model."""

import math

from jax import numpy as jnp
import haiku as hk
import distrax
from tensorflow_probability.substrates import jax as tfp

import modularbayes
from modularbayes._src.typing import Any, Array, Dict, Optional, Sequence

tfb = tfp.bijectors
tfd = tfp.distributions


class MeanField(hk.Module):
  """Auxiliary Module to assign loc and log_scale to a module.

  These parameters could be directly defined within the mean_field() function,
  but the module makes them discoverable by hk.experimental.tabulate"""

  def __init__(
      self,
      flow_dim: int,
      name: Optional[str] = None,
  ):
    super().__init__(name=name)
    self.flow_dim = flow_dim

  def __call__(self,):
    event_shape = (self.flow_dim,)

    loc = hk.get_parameter("loc", event_shape, init=jnp.zeros)
    # log_scale = jnp.zeros(event_shape)
    log_scale = hk.get_parameter("log_scale", event_shape, init=jnp.zeros)

    return loc, log_scale


def mean_field_sigma(
    num_groups: int,
    **_,
) -> modularbayes.Transformed:
  """Creates a Mean Field Flow."""

  flow_dim = num_groups  # sigma's
  event_shape = (flow_dim,)

  flow_layers = []

  # Layer 1: Trainable Affine transformation
  mf_module = MeanField(flow_dim=flow_dim)
  loc, log_scale = mf_module()
  flow_layers.append(
      distrax.Block(distrax.ScalarAffine(shift=loc, log_scale=log_scale), 1))
  # flow_layers.append(tfb.Shift(loc)(tfb.Scale(log_scale=log_scale)))

  # Last layer: Map values to parameter domain
  # sigma goes to [0,Inf]
  flow_layers.append(distrax.Block(tfb.Softplus(), 1))

  # Chain all layers together
  flow = distrax.Chain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = modularbayes.Transformed(base_distribution, flow)

  return q_distr


def mean_field_beta_tau(
    num_groups: int,
    **_,
) -> modularbayes.ConditionalTransformed:
  """Creates a Mean Field Flow."""

  flow_dim = num_groups + 1  # beta's and tau

  event_shape = (flow_dim,)

  flow_layers = []

  # Layer 1: Trainable Affine transformation
  mf_module = MeanField(flow_dim=flow_dim)
  loc, log_scale = mf_module()
  flow_layers.append(
      distrax.Block(distrax.ScalarAffine(shift=loc, log_scale=log_scale), 1))
  # flow_layers.append(tfb.Shift(loc)(tfb.Scale(log_scale=log_scale)))

  # Last layer: Map values to parameter domain
  # beta goes to [-Inf,Inf]
  # tau goes to [0,Inf]
  block_bijectors = [
      distrax.Block(tfb.Identity(), 1),
      distrax.Block(tfb.Softplus(), 1),
  ]
  block_sizes = [num_groups, 1]
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = modularbayes.ConditionalTransformed(base_distribution, flow)

  return q_distr


class CouplingConditioner(hk.Module):

  def __init__(
      self,
      output_dim: int,
      hidden_sizes: Sequence[int],
      num_bijector_params: int,
      name: Optional[str] = 'nsf_conditioner',
  ):
    super().__init__(name=name)
    self.output_dim = output_dim
    self.hidden_sizes = hidden_sizes
    self.num_bijector_params = num_bijector_params

  def __call__(self, inputs):

    out = hk.Flatten(preserve_dims=-1)(inputs)
    out = hk.nets.MLP(self.hidden_sizes, activate_final=True)(out)

    # We initialize this linear layer to zero so that the flow is initialized
    # to the identity function.
    out = hk.Linear(
        self.output_dim * self.num_bijector_params,
        w_init=jnp.zeros,
        b_init=jnp.zeros)(
            out)
    out = hk.Reshape(
        (self.output_dim,) + (self.num_bijector_params,), preserve_dims=-1)(
            out)

    return out


def nsf_sigma(
    num_groups: int,
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
    range_min: float = 0.,
    range_max: float = 1.,
    **_,
) -> modularbayes.Transformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  flow_dim = num_groups  # sigma's
  event_shape = (flow_dim,)

  flow_layers = []

  # Number of parameters required by the bijector (rational quadratic spline)
  num_bijector_params = 3 * num_bins + 1

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=range_min, range_max=range_max)

  # Alternating binary mask.
  mask = jnp.arange(0, math.prod(event_shape)) % 2
  mask = jnp.reshape(mask, event_shape)
  mask = mask.astype(bool)

  # Number of parameters for the rational-quadratic spline:
  # - `num_bins` bin widths
  # - `num_bins` bin heights
  # - `num_bins + 1` knot slopes
  # for a total of `3 * num_bins + 1` parameters.

  for _ in range(num_layers):
    layer = distrax.MaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes,
            num_bijector_params=num_bijector_params,
            name='conditioner_sigma',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # sigma goes to [0,Inf]
  flow_layers.append(distrax.Block(tfb.Softplus(), 1))

  # Chain all layers together
  flow = distrax.Chain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = modularbayes.Transformed(base_distribution, flow)

  return q_distr


def nsf_beta_tau(
    num_groups: int,
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
    range_min: float = 0.,
    range_max: float = 1.,
    **_,
) -> modularbayes.ConditionalTransformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  flow_dim = num_groups + 1  # beta's and tau

  event_shape = (flow_dim,)

  flow_layers = []

  # Number of parameters required by the bijector (rational quadratic spline)
  num_bijector_params = 3 * num_bins + 1

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=range_min, range_max=range_max)

  # Alternating binary mask.
  mask = jnp.arange(0, math.prod(event_shape)) % 2
  mask = jnp.reshape(mask, event_shape)
  mask = mask.astype(bool)

  # Number of parameters for the rational-quadratic spline:
  # - `num_bins` bin widths
  # - `num_bins` bin heights
  # - `num_bins + 1` knot slopes
  # for a total of `3 * num_bins + 1` parameters.

  for _ in range(num_layers):
    layer = modularbayes.ConditionalMaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes,
            num_bijector_params=num_bijector_params,
            name='conditioner_beta_tau',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # beta goes to [-Inf,Inf]
  # tau goes to [0,Inf]
  block_bijectors = [
      distrax.Block(tfb.Identity(), 1),
      distrax.Block(tfb.Softplus(), 1),
  ]
  block_sizes = [num_groups, 1]
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = modularbayes.ConditionalTransformed(base_distribution, flow)

  return q_distr


def meta_nsf_sigma(
    num_groups: int,
    num_layers: int,
    hidden_sizes_conditioner: Sequence[int],
    hidden_sizes_conditioner_eta: Sequence[int],
    num_bins: int,
    range_min: float = 0.,
    range_max: float = 1.,
    **_,
) -> modularbayes.ConditionalTransformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  flow_dim = num_groups  # sigma's
  event_shape = (flow_dim,)

  flow_layers = []

  # Number of parameters required by the bijector (rational quadratic spline)
  num_bijector_params = 3 * num_bins + 1

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=range_min, range_max=range_max)

  # Alternating binary mask.
  mask = jnp.arange(0, math.prod(event_shape)) % 2
  mask = jnp.reshape(mask, event_shape)
  mask = mask.astype(bool)

  # Number of parameters for the rational-quadratic spline:
  # - `num_bins` bin widths
  # - `num_bins` bin heights
  # - `num_bins + 1` knot slopes
  # for a total of `3 * num_bins + 1` parameters.

  for _ in range(num_layers):
    layer = modularbayes.EtaConditionalMaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner_eta=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner_eta,
            num_bijector_params=num_bijector_params,
            name='conditioner_eta_sigma',
        ),
        conditioner=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner,
            num_bijector_params=num_bijector_params,
            name='conditioner_sigma',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # sigma goes to [0,Inf]
  flow_layers.append(distrax.Block(tfb.Softplus(), 1))

  # Chain all layers together
  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = modularbayes.ConditionalTransformed(base_distribution, flow)

  return q_distr


def meta_nsf_beta_tau(
    num_groups: int,
    num_layers: int,
    hidden_sizes_conditioner: Sequence[int],
    hidden_sizes_conditioner_eta: Sequence[int],
    num_bins: int,
    range_min: float = 0.,
    range_max: float = 1.,
    **_,
) -> modularbayes.ConditionalTransformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  flow_dim = num_groups + 1  # beta's and tau

  event_shape = (flow_dim,)

  flow_layers = []

  # Number of parameters required by the bijector (rational quadratic spline)
  num_bijector_params = 3 * num_bins + 1

  def bijector_fn(params: Array):
    return distrax.RationalQuadraticSpline(
        params, range_min=range_min, range_max=range_max)

  # Alternating binary mask.
  mask = jnp.arange(0, math.prod(event_shape)) % 2
  mask = jnp.reshape(mask, event_shape)
  mask = mask.astype(bool)

  # Number of parameters for the rational-quadratic spline:
  # - `num_bins` bin widths
  # - `num_bins` bin heights
  # - `num_bins + 1` knot slopes
  # for a total of `3 * num_bins + 1` parameters.

  for _ in range(num_layers):
    layer = modularbayes.EtaConditionalMaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner_eta=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner_eta,
            num_bijector_params=num_bijector_params,
            name='conditioner_eta_beta_tau',
        ),
        conditioner=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner,
            num_bijector_params=num_bijector_params,
            name='conditioner_beta_tau',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # beta goes to [-Inf,Inf]
  # tau goes to [0,Inf]
  block_bijectors = [
      distrax.Block(tfb.Identity(), 1),
      distrax.Block(tfb.Softplus(), 1),
  ]
  block_sizes = [num_groups, 1]
  flow_layers.append(
      modularbayes.Blockwise(
          bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = modularbayes.ConditionalChain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = modularbayes.ConditionalTransformed(base_distribution, flow)

  return q_distr


def split_flow_sigma(
    samples: Array,
    num_groups: int,
    **_,
) -> Dict[str, Any]:
  """Get model parameters by splitting samples from the flow."""

  flow_dim = num_groups  # sigma's
  assert samples.ndim == 2
  assert samples.shape[-1] == flow_dim

  samples_dict = {}

  # sigma
  samples_dict['sigma'] = samples

  return samples_dict


def split_flow_beta_tau(
    samples: Array,
    num_groups: int,
    is_aux: bool,
    **_,
) -> Dict[str, Any]:
  """Get model parameters by splitting samples from the flow."""

  flow_dim = num_groups + 1  # beta's and tau

  assert samples.ndim == 2
  assert samples.shape[-1] == flow_dim

  samples_dict = {}

  # beta and tau
  (samples_dict['beta' + ('_aux' if is_aux else '')],
   samples_dict['tau' + ('_aux' if is_aux else '')]) = jnp.split(
       samples, [num_groups], axis=-1)

  return samples_dict
