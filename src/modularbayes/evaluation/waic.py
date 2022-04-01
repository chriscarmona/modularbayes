import warnings

import jax
from jax import numpy as jnp

import pandas as pd

from typing import Tuple

Array = jnp.ndarray


def elpd_waic(loglik_pointwise: Array) -> Array:
  """Expected log-pointwise predictive density (ELPD) via the WAIC criterion.

  Args:
    loglik_pointwise: Array of shape (s,n), where `s` is the number of samples
      from the posterior distribution and `n` is the number of data points.

  Reference:
    A. Vehtari, A. Gelman, and J. Gabry. 2017. Practical Bayesian model
    evaluation using leave-one-out cross-validation and WAIC.
    Statistics and Computing. 27(5), 1413–1432. doi:10.1007/s11222-016-9696-4
  """

  validate_ll(loglik_pointwise)

  num_samples, num_obs = loglik_pointwise.shape
  lpd = jax.scipy.special.logsumexp(
      loglik_pointwise, axis=0) - jnp.log(num_samples)  # colLogMeanExps
  p_waic = jnp.var(loglik_pointwise, axis=0)
  elpd_waic = lpd - p_waic
  waic = -2 * elpd_waic
  pointwise = jnp.stack((elpd_waic, p_waic, waic), axis=-1)

  throw_pwaic_warnings(p_waic)

  return waic_object(pointwise, dims=(num_samples, num_obs))



def validate_ll(x: Array) -> None:
  if x.ndim != 2:
    raise ValueError("Only matrices allowed in input.")
  if jnp.any(jnp.isnan(x)).item():
    raise ValueError("NAs not allowed in input.")
  if not jnp.all(jnp.isfinite(x)):
    raise ValueError("Infinite values not allowed in input.")


def throw_pwaic_warnings(p: Array, warn: bool = True):
  """ waic warnings
  Args:
      p: 'p_waic' estimates
  """
  badp = p > 0.4
  if any(badp):
    count = jnp.sum(badp)
    prop = count / len(badp)
    msg = (f"\n{count} ({100 * prop:.2f}%) p_waic estimates greater than 0.4." +
           "We recommend trying loo instead.")
    if warn:
      warnings.warn(msg)
    else:
      print(msg, "\n")


def waic_object(pointwise: Array, dims: Tuple[int]):
  estimates = table_of_estimates(pointwise)
  out = dict(estimates=estimates, pointwise=pointwise, dims=dims)
  return out


def table_of_estimates(x: Array):
  n = x.shape[0]
  out = pd.DataFrame(
      dict(Estimate=jnp.sum(x, axis=0), SE=jnp.sqrt(n * jnp.var(x, axis=0))))

  out.index = ['elpd_waic', 'p_waic', 'waic']

  return out
