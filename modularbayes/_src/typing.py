"""Pytypes for arrays and scalars."""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pathlib import Path, PosixPath

from collections import namedtuple

import numpy as np

from collections import OrderedDict
from ml_collections import ConfigDict

from tensorflow_probability.substrates import jax as tfp

from flax.training.train_state import TrainState

from chex import Array, ArrayTree, PRNGKey

ArrayNumpy = np.ndarray

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

SmiPosteriorStates = namedtuple(
    "smi_posterior_states",
    field_names=("no_cut", "cut", "cut_aux"),
)

__all__ = [
    "Any",
    "Array",
    "ArrayNumpy",
    "ArrayTree",
    "BijectorParams",
    "Callable",
    "ConfigDict",
    "Dict",
    "IntLike",
    "Iterable",
    "Iterator",
    "Kernel",
    "List",
    "Mapping",
    "Metrics",
    "NamedTuple",
    "Numeric",
    "Optional",
    "OrderedDict",
    "PRNGKey",
    "Path",
    "PosixPath",
    "RangeFloat",
    "RangeInt",
    "Sequence",
    "Scalar",
    "Shape",
    "Tuple",
    "TrainState",
    "Union",
]
