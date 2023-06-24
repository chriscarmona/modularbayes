"""modularbayes: Variational methods for Bayesian Semi-Modular Inference."""

__version__ = '0.1.4'

from modularbayes._src.bijectors.blockwise import Blockwise
from modularbayes._src.bijectors.conditional_bijector import ConditionalBijector
from modularbayes._src.bijectors.conditional_chain import ConditionalChain
from modularbayes._src.bijectors.conditional_masked_coupling import ConditionalMaskedCoupling

from modularbayes._src.conditioners.base import MeanFieldConditioner
from modularbayes._src.conditioners.base import MLPConditioner

from modularbayes._src.distributions.conditional_transformed import ConditionalTransformed
from modularbayes._src.distributions.transformed import Transformed

from modularbayes._src.metaposterior.vmp_map import MLPVmpMap

from modularbayes._src.smi.sampling import sample_q_nocut
from modularbayes._src.smi.sampling import sample_q_cutgivennocut
from modularbayes._src.smi.sampling import sample_q
from modularbayes._src.smi.elbo import elbo_smi
from modularbayes._src.smi.elbo import elbo_smi_vmpflow
from modularbayes._src.smi.elbo import elbo_smi_vmpmap

from modularbayes._src.utils.misc import cart2pol
from modularbayes._src.utils.misc import clean_filename
from modularbayes._src.utils.misc import colour_fader
from modularbayes._src.utils.misc import flatten_dict
from modularbayes._src.utils.misc import list_from_csv
from modularbayes._src.utils.misc import plot_to_image
from modularbayes._src.utils.misc import normalize_images

from modularbayes._src.utils.training import train_step

__all__ = (
    'Blockwise',
    'ConditionalBijector',
    'ConditionalChain',
    'ConditionalMaskedCoupling',
    'MeanFieldConditioner',
    'MLPConditioner',
    'ConditionalTransformed',
    'Transformed',
    'MLPVmpMap',
    'sample_q_nocut',
    'sample_q_cutgivennocut',
    'sample_q',
    'elbo_smi',
    'elbo_smi_vmpflow',
    'elbo_smi_vmpmap',
    'cart2pol',
    'clean_filename',
    'colour_fader',
    'flatten_dict',
    'list_from_csv',
    'plot_to_image',
    'normalize_images',
    'train_step',
)
