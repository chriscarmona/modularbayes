"""Miscelaneous auxiliary functions."""
import warnings
import io
import math
import unicodedata
import string

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

import csv

from jax import numpy as jnp

from modularbayes._src.typing import Array, ConfigDict, Dict, List, Tuple


def clean_filename(
    filename,
    whitelist="-_.() %s%s" % (string.ascii_letters, string.digits),
    replace=" ",
    char_limit=255,
):
  """Clean a string so it can be used for a filename.
  Source:
    https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
  """
  # replace spaces
  for r in replace:
    filename = filename.replace(r, "_")

  # keep only valid ascii chars
  cleaned_filename = (
      unicodedata.normalize("NFKD", filename).encode("ASCII",
                                                     "ignore").decode())

  # keep only whitelisted chars
  cleaned_filename = "".join(c for c in cleaned_filename if c in whitelist)
  if len(cleaned_filename) > char_limit:
    print(
        "Warning, filename truncated because it was over {}. Filenames may no longer be unique"
        .format(char_limit))
  return cleaned_filename[:char_limit]


def colour_fader(c1, c2, mix=0):
  """Linear interpolation from color c1 (at mix=0) to c2 (mix=1)

  Example
  -------

  c1='black'
  c2='gold'
  n=500

  fig, ax = plt.subplots(figsize=(8, 5))
  for x in range(n+1):
      ax.axvline(x, color=color_fader(c1,c2,x/n), linewidth=4)
  plt.show()

  """

  c1 = np.array(matplotlib.colors.to_rgb(c1))
  c2 = np.array(matplotlib.colors.to_rgb(c2))
  return matplotlib.colors.to_hex((1 - mix) * c1 + mix * c2)


def flatten_dict(input_dict, parent_key="", sep="."):
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
      v = "None"
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


def list_from_csv(file, mode="r"):
  with open(file=file, mode=mode, encoding="utf-8") as f:
    reader = csv.reader(f)
    data = list(reader)
  return data


def cart2pol(x, y):
  rho = jnp.sqrt(x**2 + y**2)
  phi = jnp.arctan2(y, x)
  phi += jnp.where(phi < 0, 2 * jnp.pi, 0.0)
  return (rho, phi)


def plot_to_image(fig):
  """Converts a matplotlib figure to a numpy array."""
  if fig is None:
    fig = plt.gcf()
  # Save the plot to a buffer
  buf = io.BytesIO()
  plt.savefig(buf, format="raw", dpi=fig.dpi)
  plt.close(fig)
  buf.seek(0)
  image = np.frombuffer(buf.getvalue(), dtype=np.uint8)
  image = np.reshape(
      image, newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
  buf.close()

  return image


def stack_images(images: Tuple[Array]):
  """Same height and width in all images"""
  h = max(x.shape[0] for x in images)
  w = max(x.shape[1] for x in images)
  for i, img in enumerate(images):
    assert img.ndim == 3
    h_pad = (h - img.shape[0]) / 2
    h_pad = math.floor(h_pad), math.ceil(h_pad)
    w_pad = (w - img.shape[1]) / 2
    w_pad = math.floor(w_pad), math.ceil(w_pad)
    images[i] = np.pad(img, (h_pad, w_pad, (0, 0)), mode="constant")
  images = np.stack(images, axis=0)
  return images


def img_stack_to_grid(
    images_array: Array,
    ncol: int,
    nrow: int,
    byrow: bool = True,
) -> Array:
  assert images_array.ndim == 4
  n_img, height, width, channel = images_array.shape
  ntotal = ncol * nrow
  if images_array.shape[0] > ntotal:
    warnings.warn(
        f"images_array contains {n_img} images, only {ntotal} are displayed.")
    images_array = images_array[:ntotal]
  else:
    # Pad to complete grid
    batch_pad = (0, ntotal - images_array.shape[0])
    images_array = np.pad(
        images_array, (batch_pad, (0, 0), (0, 0), (0, 0)), mode="constant")

  if byrow:
    reshaped_array = images_array.reshape(nrow, ncol, height, width, channel)
    reshaped_array = reshaped_array.transpose(0, 2, 1, 3, 4)
    reshaped_array = reshaped_array.reshape(nrow * height, ncol * width,
                                            channel)
  else:
    reshaped_array = images_array.reshape(nrow, ncol, height, width, channel)
    reshaped_array = reshaped_array.transpose(1, 0, 2, 3, 4)
    reshaped_array = reshaped_array.reshape(nrow * height, ncol * width,
                                            channel)

  return reshaped_array
