"""modularbayes: Variational methods for Bayesian Semi-Modular Inference."""

__version__ = '0.1.2'

from modularbayes._src.bijectors.blockwise import Blockwise
from modularbayes._src.bijectors.conditional_bijector import ConditionalBijector
from modularbayes._src.bijectors.conditional_chain import ConditionalChain
from modularbayes._src.bijectors.conditional_masked_coupling import ConditionalMaskedCoupling
from modularbayes._src.bijectors.conditional_masked_coupling_extra import EtaConditionalMaskedCoupling

from modularbayes._src.conditioners.base import MeanFieldConditioner
from modularbayes._src.conditioners.base import MLPConditioner

from modularbayes._src.distributions.conditional_transformed import ConditionalTransformed
from modularbayes._src.distributions.transformed import Transformed

from modularbayes._src.metaposterior.vmp_map import VmpMap

from modularbayes._src.utils.misc import cart2pol
from modularbayes._src.utils.misc import colour_fader
from modularbayes._src.utils.misc import flatten_dict
from modularbayes._src.utils.misc import list_from_csv
from modularbayes._src.utils.misc import plot_to_image
from modularbayes._src.utils.misc import normalize_images

from modularbayes._src.utils.training import TrainState
from modularbayes._src.utils.training import initial_state
from modularbayes._src.utils.training import initial_state_ckpt
from modularbayes._src.utils.training import load_ckpt
from modularbayes._src.utils.training import save_ckpt
from modularbayes._src.utils.training import save_checkpoint
from modularbayes._src.utils.training import update_state
from modularbayes._src.utils.training import update_states

__all__ = (
    'Blockwise',
    'ConditionalBijector',
    'ConditionalChain',
    'ConditionalMaskedCoupling',
    'EtaConditionalMaskedCoupling',
    'MeanFieldConditioner',
    'MLPConditioner',
    'ConditionalTransformed',
    'Transformed',
    'VmpMap',
    'cart2pol',
    'colour_fader',
    'flatten_dict',
    'list_from_csv',
    'plot_to_image',
    'normalize_images',
    'TrainState',
    'initial_state',
    'initial_state_ckpt',
    'load_ckpt',
    'save_ckpt',
    'save_checkpoint',
    'update_state',
    'update_states',
)
