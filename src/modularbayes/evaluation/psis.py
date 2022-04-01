# Copied and modified from:
# https://github.com/avehtari/PSIS
# Originally distributed under the GPT-3 License

from jax import numpy as jnp
from typing import Tuple

Array = jnp.ndarray


def gpdfit(x: Array) -> Tuple[Array]:
  """Estimate the paramaters for the Generalized Pareto Distribution (GPD).

  Returns empirical Bayes estimate for the parameters of the two-parameter
  generalized Parato distribution given the data.

  Args
  ----
      x: One dimensional data array

  Returns
  -------
  k, sigma : float
      estimated parameter values

  Notes
  -----
  This function returns a negative of Zhang and Stephens's k, because it is
  more common parameterisation.
  """

  if x.ndim != 1 or len(x) <= 1:
    raise ValueError("Invalid input array.")

  sort = jnp.argsort(x)

  n = len(x)
  PRIOR = 3
  m = 30 + jnp.sqrt(n).astype(int)

  bs = 1 - jnp.sqrt(m / (jnp.arange(1, m + 1, dtype=float) - 0.5))
  bs = bs / (PRIOR * x[sort[int(n / 4 + 0.5) - 1]])
  bs = bs + 1 / x[sort[-1]]

  ks = -bs
  temp = ks[:, None] * x
  temp = jnp.log1p(temp)
  ks = jnp.mean(temp, axis=1)

  L = bs / ks
  L = n * ((jnp.log(-L) - ks) - 1)

  w = 1 / jnp.sum(jnp.exp(L - L[:, None]), axis=1)

  # # remove negligible weights
  # dii = w >= 10 * jnp.finfo(float).eps
  # w = w[dii]
  # bs = bs[dii]

  # normalise w
  w = w / w.sum()

  # posterior mean for b
  b = jnp.sum(bs * w)
  # Estimate for k, note that we return a negative of Zhang and
  # Stephens's k, because it is more common parameterisation.
  k = jnp.mean(jnp.log1p((-b) * x))

  # estimate for sigma
  sigma = -k / b * n / (n - 0)
  # weakly informative prior for k
  a = 10
  k = k * n / (n + a) + a * 0.5 / (n + a)

  return k, sigma
