"""Define normalizing flows for the Epidemiology model."""

import math

from jax import numpy as jnp
import haiku as hk
import distrax
from tensorflow_probability.substrates import jax as tfp

from modularbayes import utils
from modularbayes import bijectors
from modularbayes import distributions
from modularbayes.typing import Any, Array, Dict, Optional, Sequence

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


def mean_field_phi(
    phi_dim: int,
    **_,
) -> distrax.Transformed:
  """Creates a Mean Field Flow."""

  flow_dim = phi_dim
  event_shape = (flow_dim,)

  flow_layers = []

  # Layer 1: Trainable Affine transformation
  mf_module = MeanField(flow_dim=flow_dim)
  loc, log_scale = mf_module()
  flow_layers.append(
      distrax.Block(distrax.ScalarAffine(shift=loc, log_scale=log_scale), 1))

  # Last Layer: Map values to parameter domain
  # phi goes to [0,1]
  flow_layers.append(distrax.Block(distrax.Sigmoid(), 1))

  # Chain all layers together
  flow = distrax.Chain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = distributions.Transformed(base_distribution, flow)

  return q_distr


def mean_field_theta(
    theta_dim: int,
    **_,
) -> distrax.Transformed:
  """Creates a Mean Field Flow."""

  flow_dim = theta_dim

  event_shape = (flow_dim,)

  flow_layers = []

  # Layer 1: Trainable Affine transformation
  mf_module = MeanField(flow_dim=flow_dim)
  loc, log_scale = mf_module()
  flow_layers.append(
      distrax.Block(distrax.ScalarAffine(shift=loc, log_scale=log_scale), 1))
  # flow_layers.append(tfb.Shift(loc)(tfb.Scale(log_scale=log_scale)))

  # Last layer: Map values to parameter domain
  # theta1 goes to [-Inf,Inf]
  # theta2 goes to [0,Inf]
  block_bijectors = [
      distrax.Block(tfb.Identity(), 1),
      distrax.Block(tfb.Softplus(), 1),
  ]
  block_sizes = [
      1,
      1,
  ]
  flow_layers.append(
      bijectors.Blockwise(bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = bijectors.ConditionalChain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = distributions.ConditionalTransformed(base_distribution, flow)

  return q_distr


def gaussian(
    phi_dim: int,
    theta_dim: int,
    is_smi: bool,
    **_,
) -> distrax.Transformed:
  """Creates a Mean Field Flow."""
  # event_shape = (15,)

  shared_params_dim = phi_dim
  refit_params_dim = theta_dim

  flow_dim = shared_params_dim + (2 if is_smi else 1) * refit_params_dim

  event_shape = (flow_dim,)

  flow_layers = []

  # Layer 1: Scale by cov_chol, a lower triangular matrix
  _pre_cov_chol = hk.get_parameter(
      "_pre_cov_chol", event_shape * 2, init=jnp.zeros)
  cov_chol = utils.as_lower_chol(_pre_cov_chol)
  flow_layers.append(tfb.ScaleMatvecTriL(scale_tril=cov_chol))

  # Layer 2: Shift by loc
  loc = hk.get_parameter("loc", event_shape, init=jnp.zeros)
  flow_layers.append(distrax.Block(tfb.Shift(shift=loc), 1))

  # Layer 3: Map values to parameter domain
  # phi goes to [0,1]
  # theta1 goes to [-Inf,Inf]
  # theta2 goes to [0,Inf]
  block_bijectors = [distrax.Block(distrax.Sigmoid(), 1)
                    ] + (2 if is_smi else 1) * [
                        distrax.Block(tfb.Identity(), 1),
                        distrax.Block(tfb.Softplus(), 1)
                    ]
  block_sizes = [phi_dim] + (2 if is_smi else 1) * [1, 1]
  flow_layers.append(
      bijectors.Blockwise(bijectors=block_bijectors, block_sizes=block_sizes))

  # Chain all layers together
  flow = distrax.Chain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = distrax.Transformed(base_distribution, flow)

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


def nsf_phi(
    phi_dim: int,
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
    range_min: float = 0.,
    range_max: float = 1.,
    **_,
) -> distrax.Transformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  flow_dim = phi_dim

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
            name='conditioner_phi',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # phi goes to [0,1]
  flow_layers.append(distrax.Block(distrax.Sigmoid(), 1))

  flow = distrax.Chain(flow_layers[::-1])

  # base_distribution = distrax.Independent(
  #     distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
  #     reinterpreted_batch_ndims=len(event_shape))

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  return distributions.Transformed(base_distribution, flow)


def nsf_theta(
    theta_dim: int,
    num_layers: int,
    hidden_sizes: Sequence[int],
    num_bins: int,
    range_min: float = 0.,
    range_max: float = 1.,
    **_,
) -> distrax.Transformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  flow_dim = theta_dim
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
    layer = bijectors.ConditionalMaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner=CouplingConditioner(
            # input_dim=math.prod(event_shape),
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes,
            num_bijector_params=num_bijector_params,
            name='conditioner_theta',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # theta1 goes to [-Inf,Inf]
  # theta2 goes to [0,Inf]
  block_bijectors = [
      distrax.Block(tfb.Identity(), 1),
      distrax.Block(tfb.Softplus(), 1),
  ]
  block_sizes = [
      1,
      1,
  ]
  flow_layers.append(
      bijectors.Blockwise(bijectors=block_bijectors, block_sizes=block_sizes))
  flow = bijectors.ConditionalChain(flow_layers[::-1])

  # base_distribution = distrax.Independent(
  #     distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
  #     reinterpreted_batch_ndims=len(event_shape))

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  return distributions.ConditionalTransformed(base_distribution, flow)


def meta_nsf_phi(
    phi_dim: int,
    num_layers: int,
    hidden_sizes_conditioner: Sequence[int],
    hidden_sizes_conditioner_eta: Sequence[int],
    num_bins: int,
    range_min: float = 0.,
    range_max: float = 1.,
    **_,
) -> distributions.ConditionalTransformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  flow_dim = phi_dim

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
    layer = bijectors.EtaConditionalMaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner_eta=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner_eta,
            num_bijector_params=num_bijector_params,
            name='conditioner_eta_phi',
        ),
        conditioner=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner,
            num_bijector_params=num_bijector_params,
            name='conditioner_phi',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # phi goes to [0,1]
  flow_layers.append(distrax.Block(distrax.Sigmoid(), 1))

  flow = bijectors.ConditionalChain(flow_layers[::-1])

  # base_distribution = distrax.Independent(
  #     distrax.Uniform(low=jnp.zeros(event_shape), high=jnp.ones(event_shape)),
  #     reinterpreted_batch_ndims=len(event_shape))

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  return distributions.ConditionalTransformed(base_distribution, flow)


def meta_nsf_theta(
    theta_dim: int,
    num_layers: int,
    hidden_sizes_conditioner: Sequence[int],
    hidden_sizes_conditioner_eta: Sequence[int],
    num_bins: int,
    range_min: float = 0.,
    range_max: float = 1.,
    **_,
) -> distributions.ConditionalTransformed:
  """Creates the Rational Quadratic Flow model.

  Args:
  range_min: the lower bound of the spline's range. Below `range_min`, the
    bijector defaults to a linear transformation.
  range_max: the upper bound of the spline's range. Above `range_max`, the
    bijector defaults to a linear transformation.
  """

  flow_dim = theta_dim
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
    layer = bijectors.EtaConditionalMaskedCoupling(
        mask=mask,
        bijector=bijector_fn,
        conditioner_eta=CouplingConditioner(
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner_eta,
            num_bijector_params=num_bijector_params,
            name='conditioner_eta_theta',
        ),
        conditioner=CouplingConditioner(
            # input_dim=math.prod(event_shape),
            output_dim=math.prod(event_shape),
            hidden_sizes=hidden_sizes_conditioner,
            num_bijector_params=num_bijector_params,
            name='conditioner_theta',
        ),
    )
    flow_layers.append(layer)
    # Flip the mask after each layer.
    mask = jnp.logical_not(mask)

  # Last layer: Map values to parameter domain
  # theta1 goes to [-Inf,Inf]
  # theta2 goes to [0,Inf]
  block_bijectors = [
      distrax.Block(tfb.Identity(), 1),
      distrax.Block(tfb.Softplus(), 1),
  ]
  block_sizes = [
      1,
      1,
  ]
  flow_layers.append(
      bijectors.Blockwise(bijectors=block_bijectors, block_sizes=block_sizes))
  flow = bijectors.ConditionalChain(flow_layers[::-1])

  base_distribution = distrax.MultivariateNormalDiag(
      loc=jnp.zeros(event_shape), scale_diag=jnp.ones(event_shape))

  q_distr = distributions.ConditionalTransformed(base_distribution, flow)

  return q_distr


def split_flow_phi(
    samples: Array,
    phi_dim: int,
    **_,
) -> Dict[str, Any]:
  """Get model parameters by splitting samples from the flow."""

  flow_dim = phi_dim

  assert samples.ndim == 2
  assert samples.shape[-1] == flow_dim

  samples_dict = {}

  # phi: Human-Papilloma virus (HPV) prevalence on each population
  samples_dict['phi'] = samples

  return samples_dict


def split_flow_theta(
    samples: Array,
    theta_dim: int,
    is_aux: bool,
    **_,
) -> Dict[str, Any]:
  """Get model parameters by splitting samples from the flow."""

  flow_dim = theta_dim

  assert samples.ndim == 2
  assert samples.shape[-1] == flow_dim

  samples_dict = {}

  # theta: Intercept and slope of the prevalence-incidence model
  samples_dict['theta' + ('_aux' if is_aux else '')] = samples

  return samples_dict
