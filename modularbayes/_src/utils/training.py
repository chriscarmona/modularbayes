"""A simple example of a flow model trained on epidemiology data."""
import functools
import os
import pathlib
import pickle

from absl import logging

from tensorflow.io import gfile

import haiku as hk
import jax
from jax import lax
from jax import numpy as jnp

import optax

from modularbayes._src.typing import (Any, Batch, Callable, Dict, List, Metrics,
                                 NamedTuple, PRNGKey, Tuple, Union)


def save_ckpt(state: NamedTuple, path: Union[str, pathlib.Path]) -> None:
  r"""Save Haiku state.

    A more mature checkpointing implementation might:
    - Use np.savez() to store the core data instead of pickle.
    - Automatically garbage collect old checkpoints.
  """
  checkpoint_state = jax.device_get(state)
  logging.info('Serializing experiment state to %s', path)
  with open(path, 'wb') as f:
    pickle.dump(checkpoint_state, f)


def load_ckpt(path: Union[str, pathlib.Path]) -> NamedTuple:
  r"""Load Haiku state.

    A more mature checkpointing implementation might:
    - Use np.savez() to store the core data instead of pickle.
    - Automatically garbage collect old checkpoints.
    """
  logging.info('Loading checkpoint from %s', path)
  with open(path, 'rb') as f:
    state = pickle.load(f)
  return state


class TrainState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState
  step: int


def initial_state(
    forward_fn: hk.Transformed,
    forward_fn_kwargs: Dict[str, Any],
    prng_key: PRNGKey,
    optimizer: optax.GradientTransformation,
) -> TrainState:
  """Create the initial network state."""
  params = forward_fn.init(prng_key, **forward_fn_kwargs)
  opt_state = optimizer.init(params)
  return TrainState(params, opt_state, 0)


def initial_state_ckpt(
    checkpoint_dir: str,
    forward_fn: hk.Transformed,
    forward_fn_kwargs: Dict[str, Any],
    prng_key: PRNGKey,
    optimizer: optax.GradientTransformation,
) -> TrainState:
  """Create the initial network state, considering existing checkpoints."""
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
  checkpoint_files = [p for p in os.listdir(checkpoint_dir) if 'ckpt_' in p]
  checkpoint_files.sort()

  if not checkpoint_files:
    state = initial_state(
        forward_fn=forward_fn,
        forward_fn_kwargs=forward_fn_kwargs,
        prng_key=prng_key,
        optimizer=optimizer,
    )
  else:
    state = load_ckpt(path=pathlib.Path(checkpoint_dir) / checkpoint_files[-1])

  return state


def save_checkpoint(
    state: TrainState,
    checkpoint_dir: str,
    keep: int = 1,
    overwrite: bool = False,
) -> None:
  """Write the provided training state to disk."""
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(state)
    step = int(state.step)

    checkpoint_files = [p for p in os.listdir(checkpoint_dir) if 'ckpt_' in p]
    checkpoint_files.sort()

    checkpoint_file_now = f'ckpt_{step:06d}'
    if checkpoint_file_now not in checkpoint_files:
      checkpoint_files.append(checkpoint_file_now)
      save_ckpt(state, str(pathlib.Path(checkpoint_dir) / checkpoint_file_now))
    elif overwrite:
      save_ckpt(state, str(pathlib.Path(checkpoint_dir) / checkpoint_file_now))

    if len(checkpoint_files) > keep:
      old_ckpts = checkpoint_files[:-keep]
      for old_ckpt in old_ckpts:
        ckpt_path = pathlib.Path(checkpoint_dir) / old_ckpt
        logging.info('Removing checkpoint at %s', ckpt_path)
        gfile.remove(ckpt_path)


def update_state(
    state: TrainState,
    batch: Batch,
    prng_key: PRNGKey,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable,
    loss_fn_kwargs: Dict[str, Any],
) -> Tuple[TrainState, Metrics]:
  """Single SGD update step."""

  params, opt_state, step = state

  train_loss, grads = jax.value_and_grad(loss_fn)(
      params, batch=batch, prng_key=prng_key, **loss_fn_kwargs)

  metrics = {'train_loss': train_loss}

  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)

  # Define new train state

  # new_state = TrainState(
  #     new_params,
  #     new_opt_state,
  #     step + 1,
  # )

  # Check that all gradients are finite
  finite = jnp.array(True)
  for g in jax.tree_leaves(grads):
    finite &= jnp.all(lax.is_finite(g))

  # If the gradients are not finite, keep previous params and opt_state
  new_state = TrainState(
      jax.tree_map(functools.partial(jnp.where, finite), new_params, params),
      jax.tree_map(
          functools.partial(jnp.where, finite), new_opt_state, opt_state),
      step + 1,
  )

  return new_state, metrics


def update_states(
    state_list: List[TrainState],
    batch: Batch,
    prng_key: PRNGKey,
    optimizer: optax.GradientTransformation,
    loss_fn: Callable,
    loss_fn_kwargs: Dict[str, Any],
) -> Tuple[List[TrainState], Metrics]:
  """Single SGD update step."""

  params_tuple, opt_state_tuple, step_tuple = zip(*state_list)

  train_loss, grads = jax.value_and_grad(loss_fn)(
      params_tuple, batch=batch, prng_key=prng_key, **loss_fn_kwargs)

  metrics = {'train_loss': train_loss}

  new_state_list = []
  for i in range(len(state_list)):
    updates, new_opt_state = optimizer.update(grads[i], opt_state_tuple[i])
    new_params = optax.apply_updates(params_tuple[i], updates)

    # Check that all gradients are finite
    finite = jnp.array(True)
    for g in jax.tree_leaves(grads):
      finite &= jnp.all(jax.lax.is_finite(g))

    # If the gradients are not finite, keep previous params and opt_state
    new_state_list.append(
        TrainState(
            params=jax.tree_map(
                functools.partial(jnp.where, finite),
                new_params,
                params_tuple[i],
            ),
            opt_state=jax.tree_map(
                functools.partial(jnp.where, finite),
                new_opt_state,
                opt_state_tuple[i],
            ),
            step=step_tuple[i] + 1,
        ))

  return new_state_list, metrics
