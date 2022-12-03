"""Define Cubic Spline map for Variational Meta-Posteriors"""

import math

import haiku as hk
import jax
from jax import numpy as jnp

from modularbayes._src.typing import Array, List, Optional


class VmpMap(hk.Module):
  """Trainable mapping from eta to normalizing flow parameters."""

  def __init__(
      self,
      params_flow_init: hk.Params,
      hidden_sizes: List[int],
      name: Optional[str] = None,
  ):
    super().__init__(name=name)

    # Variational parameters to be produced
    # Tree definition
    leaves, self.params_flow_treedef = jax.tree_util.tree_flatten(
        params_flow_init)
    self.params_flow_shapes = [x.shape for x in leaves]
    # Total number of output paramemters by the vmp
    self.output_dim = sum(x.flatten().shape[0] for x in leaves)

    self.hidden_sizes = hidden_sizes

  def __call__(self, eta: Array) -> List[hk.Params]:
    assert eta.ndim == 2

    num_eta_new, _ = eta.shape

    vmp_map = hk.Sequential([
        hk.Flatten(preserve_dims=-1),
        hk.nets.MLP(
            output_sizes=self.hidden_sizes,
            activation=jax.nn.leaky_relu,
            activate_final=True,
        ),
        hk.Linear(output_size=self.output_dim),
    ])
    flow_params_merged = jax.vmap(vmp_map)(eta)

    # out1 = hk.Flatten(preserve_dims=-1)(eta)
    # out1 = hk.nets.MLP(
    #     self.hidden_sizes,
    #     activation=jax.nn.leaky_relu,
    #     activate_final=True,
    # )(
    #     out1)
    # out2 = hk.Flatten(preserve_dims=-1)(eta)
    # out2 = hk.nets.MLP(
    #     self.hidden_sizes[-1:],
    #     activation=jax.nn.leaky_relu,
    #     activate_final=True,
    # )(
    #     out2)
    # flow_params_merged = hk.Linear(output_size=self.output_dim)(out1 + out2)

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
