"""Main script for running the epidemiology model."""

from absl import app
from absl import flags
from absl import logging

from clu import platform

import jax
from ml_collections import config_flags
import tensorflow as tf

import train_flow
import train_vmp_map
import run_mcmc

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  tf.config.experimental.set_visible_devices([], 'TPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  logging.info('JAX device count: %r', jax.device_count())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')
  if FLAGS.config.method == 'mcmc':
    if FLAGS.config.iterate_smi_eta == ():
      FLAGS.config.smi_eta = None
      run_mcmc.sample_and_evaluate(FLAGS.config, FLAGS.workdir)
    else:
      for eta in FLAGS.config.iterate_smi_eta:
        FLAGS.config.smi_eta = {'modules': [[1.0, eta]]}
        run_mcmc.sample_and_evaluate(FLAGS.config,
                                     FLAGS.workdir + f"_{eta:.3f}")
  elif FLAGS.config.method == 'flow':
    if FLAGS.config.iterate_smi_eta == ():
      FLAGS.config.flow_kwargs.smi_eta = None
      train_flow.train_and_evaluate(FLAGS.config, FLAGS.workdir)
    else:
      for eta in FLAGS.config.iterate_smi_eta:
        FLAGS.config.flow_kwargs.smi_eta = {'modules': [[1.0, eta]]}

        train_flow.train_and_evaluate(FLAGS.config,
                                      FLAGS.workdir + f"_{eta:.3f}")

  elif FLAGS.config.method == 'vmp_map':
    train_vmp_map.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
