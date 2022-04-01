"""Pytypes for arrays and scalars."""

from typing import (Any, Callable, Dict, Iterable, Iterator, List, Mapping,
                    NamedTuple, Optional, Sequence, Tuple, Union)

from pathlib import Path, PosixPath

import jax
import jax.numpy as jnp

import numpy as np

from flax.metrics.tensorboard import SummaryWriter

from chex import Array, PRNGKey

from ml_collections import ConfigDict

from tensorflow_probability.substrates import jax as tfp

ArrayNumpy = np.ndarray
ArraySharded = jax.interpreters.pxla.ShardedDeviceArray

BijectorParams = Any

Scalar = Union[float, int]
Numeric = Union[Array, Scalar]
Shape = Tuple[int, ...]

IntLike = Union[int, np.int16, np.int32, np.int64]

Metrics = Mapping[str, float]
Batch = Mapping[str, ArrayNumpy]

RangeFloat = Tuple[float, float]
RangeInt = Tuple[int, Union[int, None]]

Kernel = tfp.math.psd_kernels.PositiveSemidefiniteKernel

SmiEta = Mapping[str, np.ndarray]
