"""Define Trainable Conditioner to use in the Normalizing Flows.

The transformation in a normalizing flow can be interpreted of two parts:
1) Conditioner: Takes the input epsilon and produce parameters that will be
     used by the bijector. The parameters in a Normalising flow are often here
     Eg. in an affine AR flow, take epsilon_{1:t-1} to produce loc_t and
     log_scale_t.
2) Bijector: Transform epsilon using a function that depend on the parameters
     produced by the conditioner.
     Eg. Affine transformation: epsilon_t * scale_t + loc_t

In this script we define functions that can be used as conditioners.
"""

from jax import numpy as jnp
import haiku as hk

from modularbayes._src.typing import Optional, Sequence


class MeanFieldConditioner(hk.Module):
  """Mean Field Conditioner.

  This is an auxiliary conditioner that does not take any input, simply returns
  `loc` and `log_scale` to be used by an affine bijector.
  By assigning these these parameters to a hk.Module, they are also discoverable
  by hk.experimental.tabulate.
  """

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


class MLPConditioner(hk.Module):
  """Multi-Layer Perceptron (MLP) Conditioner.
  This conditioner takes the flow input, pass it through a MLP and produce
  parameters for the bijector as required. This is normally used in with the
  Masked Coupling Bijector to make sure that the lower triangular Jabobian is
  preserved.
  """

  def __init__(
      self,
      output_dim: int,
      hidden_sizes: Sequence[int],
      num_bijector_params: int,
      name: Optional[str] = "nsf_conditioner",
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
