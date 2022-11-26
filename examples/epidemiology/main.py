"""Main script for training the model."""

import os
import warnings

from absl import app
from absl import flags
from absl import logging

import jax
from ml_collections import config_flags
import tensorflow as tf

import run_mcmc
import train_flow
import train_vmp_flow
import train_vmp_map

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=False)

# TODO: Remove when Haiku stop producing "jax.tree_leaves is deprecated" warning
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # log to a file
  if FLAGS.log_dir:
    if not os.path.exists(FLAGS.log_dir):
      os.makedirs(FLAGS.log_dir)
    logging.get_absl_handler().use_absl_log_file()

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  tf.config.experimental.set_visible_devices([], 'TPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  logging.info('JAX device count: %r', jax.device_count())

  if FLAGS.config.method == 'mcmc':
    run_mcmc.sample_and_evaluate(FLAGS.config, FLAGS.workdir)
  elif FLAGS.config.method == 'flow':
    train_flow.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  elif FLAGS.config.method == 'vmp_flow':
    train_vmp_flow.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  elif FLAGS.config.method == 'vmp_map':
    train_vmp_map.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
