"""A simple example of a flow model trained on epidemiology data."""

import jax

from flax.training.train_state import TrainState

from modularbayes._src.typing import Any, Callable, Dict, PRNGKey, Tuple


def train_step(
    state_tuple: Tuple[TrainState],
    batch: Dict[str, Any],
    prng_key: PRNGKey,
    loss: Callable,
    loss_kwargs: Dict[str, Any],
):
  """Generic SGD step."""
  params_tuple = tuple(x.params for x in state_tuple)
  grads_tuple = jax.grad(loss)(
      params_tuple, prng_key=prng_key, batch=batch, **loss_kwargs)
  new_state_tuple = [
      state.apply_gradients(grads=grads)
      for state, grads in zip(state_tuple, grads_tuple)
  ]
  new_train_loss = loss(
      tuple(x.params for x in new_state_tuple),
      prng_key=prng_key,
      batch=batch,
      **loss_kwargs)
  metrics = {'train_loss': new_train_loss}
  return new_state_tuple, metrics
