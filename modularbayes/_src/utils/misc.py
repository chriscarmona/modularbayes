import io
import math

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

import tensorflow as tf
import csv

import jax
from jax import numpy as jnp
from jax import lax

from modularbayes._src.typing import Array, ConfigDict, Dict, List, Optional, Tuple


def as_lower_chol(x: Array) -> Array:
  """Create a matrix that could be used as Lower Cholesky.

    Args:
      x: Square matrix.

    Returns:
      Lower triangular matrix with positive diagonal.

    Note:
      The function simply masks for the upper triangular part of the matrix
      followed by an exponential transformation to the diagonal, in order to make it positive.
      Does not calculate the cholesky decomposition.
  """
  return jnp.tril(x, -1) + jnp.diag(jax.nn.softplus(jnp.diag(x)))


def colour_fader(c1, c2, mix=0):
  '''fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)

    Example
    -------

    c1='black'
    c2='gold'
    n=500

    fig, ax = plt.subplots(figsize=(8, 5))
    for x in range(n+1):
        ax.axvline(x, color=color_fader(c1,c2,x/n), linewidth=4)
    plt.show()

    '''

  c1 = np.array(matplotlib.colors.to_rgb(c1))
  c2 = np.array(matplotlib.colors.to_rgb(c2))
  return matplotlib.colors.to_hex((1 - mix) * c1 + mix * c2)


def issymmetric(a, rtol=1e-05, atol=1e-08):
  return jnp.allclose(a, a.T, rtol=rtol, atol=atol)


def cholesky_expand_right(m: Array, L: Array, m_new: Array):
  """Cholesky Factor Update to append new rows/columns.

    Args
    ____
    m : Original matrix to be expanded.
    L : Lower triangular matrix with cholesky factor of the original matrix `m`.
    m_new : New matrix after adding `d` rows and columns to m

    Reference
    ---------
    Osborne, M. A. (2010). Bayesian Gaussian processes for sequential
      prediction, optimisation and quadrature [PhD thesis]. Appendix B. Oxford
      University, UK.
    """

  # Original dimension
  d = m.shape[0]
  # Number of new rows/columns
  k = m_new.shape[0] - d

  R11 = L.transpose()

  S11 = R11
  S12 = jax.scipy.linalg.solve_triangular(a=R11, b=m_new[:d, d:], trans=1)
  S22 = jnp.linalg.cholesky(m_new[d:, d:] - S12.transpose() @ S12).transpose()

  L_new = jnp.concatenate([
      jnp.concatenate([S11, S12], axis=1),
      jnp.concatenate([jnp.zeros((k, d)), S22], axis=1)
  ],
                          axis=0).transpose()

  return L_new


def flatten_dict(input_dict, parent_key='', sep='.'):
  """Flattens and simplifies dict such that it can be used by hparams.
  Args:
    input_dict: Input dict, e.g., from ConfigDict.
    parent_key: String used in recursion.
    sep: String used to separate parent and child keys.
  Returns:
   Flattened dict.
  """
  items = []
  for k, v in input_dict.items():
    new_key = parent_key + sep + k if parent_key else k

    # Take special care of things hparams cannot handle.
    if v is None:
      v = 'None'
    elif isinstance(v, List):
      v = str(v)
    elif isinstance(v, Tuple):
      v = str(v)
    elif isinstance(v, Dict) or isinstance(v, ConfigDict):
      # Recursively flatten the dict.
      items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)


def force_symmetric(A, lower=True):
  """Create a symmetric matrix by copying lower/upper diagonal"""
  A_tri = jnp.tril(A) if lower else jnp.triu(A)
  return A_tri + A_tri.T - jnp.diag(jnp.diag(A_tri))


def list_from_csv(file, mode='r'):
  with open(file, mode) as f:
    reader = csv.reader(f)
    data = list(reader)
  return data


def log1mexpm(x):
  """Accurately Computes log(1 - exp(-x)).

  Source:
    https://cran.r-project.org/web/packages/Rmpfr/
  """

  return jnp.log(-jnp.expm1(-x))


def cart2pol(x, y):
  rho = jnp.sqrt(x**2 + y**2)
  phi = jnp.arctan2(y, x)
  phi += jnp.where(phi < 0, 2 * jnp.pi, 0.)
  return (rho, phi)


def _square_scaled_dist(x1: Array,
                        x2: Optional[Array] = None,
                        lengthscale: float = 1.0) -> Array:
  """Returns :math:`\|\frac{x1-x2}{l}\|^2`."""
  if x2 is None:
    x2 = x1
  if x1.shape[-1] != x2.shape[-1]:
    raise ValueError("Inputs must have the same number of features.")

  scaled_x1 = x1 / lengthscale
  scaled_x2 = x2 / lengthscale
  x1_sq = (scaled_x1**2).sum(axis=-1, keepdims=True)
  x2_sq = (scaled_x2**2).sum(axis=-1, keepdims=True)
  x1x2 = scaled_x1 @ scaled_x2.T
  r2 = x1_sq - 2 * x1x2 + x2_sq.T
  return lax.clamp(min=0., x=r2, max=jnp.inf)


def plot_to_image(fig):
  """Converts the matplotlib plot to a PNG image.

  The supplied figure is closed and inaccessible after this call.
  """
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(fig)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def normalize_images(images):
  """Same height and width in all images"""
  h, w = max([x.shape[1] for x in images]), max([x.shape[2] for x in images])
  for i, img in enumerate(images):
    h_pad = (h - img.shape[1]) / 2
    h_pad = math.floor(h_pad), math.ceil(h_pad)
    w_pad = (w - img.shape[2]) / 2
    w_pad = math.floor(w_pad), math.ceil(w_pad)
    images[i] = np.pad(img, ((0, 0), h_pad, w_pad, (0, 0)), mode='constant')

  images = np.concatenate(images, axis=0)

  return images
