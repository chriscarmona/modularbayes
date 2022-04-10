"""Maximul Likelihood Estimation (MLE) for the Epidemiology example."""

from typing import NamedTuple
import numpy as np

import jax
from jax import numpy as jnp
import haiku as hk
import optax

import data
import log_prob_fun

np.set_printoptions(suppress=True, precision=4)

if __name__ == '__main__':
  phi_dim = 13
  theta_dim = 2
  model_params_dim = phi_dim + theta_dim
  model_params_split = jnp.array(np.cumsum([phi_dim, int(theta_dim / 2)]))

  prng_seq = hk.PRNGSequence(123)

  learning_rate = 1e-1
  training_steps = 1000
  eval_steps = int(training_steps / 10)

  train_ds = dict(
      zip(['Z', 'N', 'Y', 'T'],
          jnp.split(data.epidemiology.to_numpy(), 4, axis=-1)))
  train_ds = {key: value.squeeze() for key, value in train_ds.items()}

  @hk.without_apply_rng
  @hk.transform
  def loglik(batch):
    _loc = hk.get_parameter("_loc", (1, model_params_dim), init=jnp.zeros)
    # _loc = jnp.zeros((1, model_params_dim))

    _model_params = jnp.split(_loc, model_params_split, axis=-1)
    model_params = jnp.concatenate(
        [
            jax.nn.sigmoid(_model_params[0]),  # phi
            _model_params[1],  # theta1
            jnp.exp(_model_params[2]),  # theta2
        ],
        axis=-1)
    phi, theta = jnp.split(model_params, [phi_dim], axis=-1)

    return log_prob_fun.log_lik_vectorised(phi=phi, theta=theta, **batch)

  @hk.without_apply_rng
  @hk.transform
  def get_model_params():
    """Compute log_prob."""

    _loc = hk.get_parameter("_loc", (1, model_params_dim), init=jnp.zeros)
    # _loc = jnp.zeros((1, model_params_dim))

    _model_params = jnp.split(_loc, model_params_split, axis=-1)
    model_params = jnp.concatenate(
        [
            jax.nn.sigmoid(_model_params[0]),  # phi
            _model_params[1],  # theta1
            jnp.exp(_model_params[2]),  # theta2
        ],
        axis=-1)

    return model_params

  loss_fn = lambda params, batch: -loglik.apply(params, batch=batch).sum()

  class TrainState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState
    step: int

  def make_optimizer(lr) -> optax.GradientTransformation:
    return optax.adam(lr)

  # Define initial state
  def initial_state():
    params = loglik.init(next(prng_seq), batch=train_ds)
    loglik.apply(params, batch=train_ds)
    opt_state = make_optimizer(learning_rate).init(params)
    return TrainState(params, opt_state, 0)

  train_state = initial_state()

  @jax.jit
  def update(state, batch):
    """Single SGD update step."""

    params, opt_state, step = state
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, new_opt_state = make_optimizer(learning_rate).update(
        grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    new_state = TrainState(new_params, new_opt_state, step + 1)

    metrics = {
        'loss': loss,
    }

    return new_state, metrics

  pbar = range(int(training_steps) + 1)
  for train_step in pbar:
    # step=0
    train_state, train_metrics = update(state=train_state, batch=train_ds)
    if (train_step + 1) % eval_steps == 0:
      print(f"STEP: {train_step}; training loss: {train_metrics['loss']:.3f}")

  # get_model_params.apply(train_state.params)
  # loglik.apply(train_state.params, batch=data)[0, :, 0]
  # data["Z"] / data["N"]
