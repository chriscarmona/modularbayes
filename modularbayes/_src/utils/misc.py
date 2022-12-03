"""Miscelaneous auxiliary functions."""

import io
import math

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

import tensorflow as tf
import csv

from jax import numpy as jnp

from modularbayes._src.typing import ConfigDict, Dict, List, Tuple


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
    elif isinstance(v, (ConfigDict, Dict)):
      # Recursively flatten the dict.
      items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)


def list_from_csv(file, mode='r'):
  with open(file=file, mode=mode, encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)
  return data


def cart2pol(x, y):
  rho = jnp.sqrt(x**2 + y**2)
  phi = jnp.arctan2(y, x)
  phi += jnp.where(phi < 0, 2 * jnp.pi, 0.)
  return (rho, phi)


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
  h = max(x.shape[1] for x in images)
  w = max(x.shape[2] for x in images)
  for i, img in enumerate(images):
    h_pad = (h - img.shape[1]) / 2
    h_pad = math.floor(h_pad), math.ceil(h_pad)
    w_pad = (w - img.shape[2]) / 2
    w_pad = math.floor(w_pad), math.ceil(w_pad)
    images[i] = np.pad(img, ((0, 0), h_pad, w_pad, (0, 0)), mode='constant')

  images = np.concatenate(images, axis=0)

  return images
