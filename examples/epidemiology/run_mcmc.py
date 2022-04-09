"""MCMC sampling for the Epidemiology model."""
from typing import Any, Dict, Mapping, Optional

import time
from absl import logging

import ml_collections

import numpy as np

import jax
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp

from flax.metrics import tensorboard

import haiku as hk

import log_prob_fun
import plot

import modularbayes
from modularbayes import utils

tfd = tfp.distributions
tfb = tfp.bijectors
tfm = tfp.mcmc

np.set_printoptions(suppress=True, precision=4)

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
ConfigDict = ml_collections.ConfigDict
SmiEta = Mapping[str, np.ndarray]


def load_dataset() -> Dict[str, Array]:
  """Load Epidemiology data from Plummer."""
  data = dict(
      zip(['Z', 'N', 'Y', 'T'],
          np.split(modularbayes.data.epidemiology.to_numpy(), 4, axis=-1)))
  data = {key: jnp.array(value.squeeze()) for key, value in data.items()}
  return data


@jax.jit
def log_prob_fn(
    batch: Batch,
    model_params: Array,
    smi_eta_modules: Optional[Array],
    model_params_init: Mapping[str, Any],
):
  """Log probability function for the Epidemiology model."""

  leaves_init, treedef = jax.tree_util.tree_flatten(model_params_init)

  leaves = []
  for i in range(len(leaves_init) - 1):
    param_i, model_params = jnp.split(
        model_params, leaves_init[i].flatten().shape, axis=-1)
    leaves.append(param_i.reshape(leaves_init[i].shape))
  leaves.append(model_params.reshape(leaves_init[-1].shape))

  posterior_sample_dict = jax.tree_util.tree_unflatten(
      treedef=treedef, leaves=leaves)

  # # Bijector
  # posterior_sample_dict['phi'] = jax.nn.sigmoid(posterior_sample_dict['phi'])
  # theta1, theta2_ = jnp.split(posterior_sample_dict['theta'], 2, axis=-1)
  # theta2 = jax.nn.softplus(theta2_)
  # posterior_sample_dict['theta'] = jnp.concatenate([theta1, theta2], axis=-1)

  smi_eta = {
      'modules': smi_eta_modules
  } if smi_eta_modules is not None else None

  log_prob = log_prob_fun.log_prob_joint(
      batch=batch,
      posterior_sample_dict=posterior_sample_dict,
      smi_eta=smi_eta,
  ).squeeze()

  return log_prob


def sample_and_evaluate(config: ConfigDict, workdir: str) -> Mapping[str, Any]:
  """Sample from the posterior in the Epidemiology model."""

  # Full dataset used everytime
  # Small data, no need to batch
  train_ds = load_dataset()

  # Initialize random keys
  prng_seq = hk.PRNGSequence(config.seed)

  phi_dim = train_ds['Z'].shape[0]
  theta_dim = 2

  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(utils.flatten_dict(config))

  smi_eta = config.smi_eta

  posterior_sample_dict_init = {}
  posterior_sample_dict_init['phi'] = 0.2 * jnp.ones((1, phi_dim))
  # posterior_sample_dict_init['theta'] = jnp.ones((1, 2))
  posterior_sample_dict_init['theta'] = jnp.array([[-1.5, 20]])

  ### Sample First Stage ###

  logging.info("\t sampling stage 1...")

  times_data = {}
  times_data['start_sampling'] = time.perf_counter()

  target_log_prob_fn = lambda state: log_prob_fn(
      batch=train_ds,
      model_params=state,
      smi_eta_modules=jnp.array(smi_eta['modules'])
      if smi_eta is not None else None,
      model_params_init=posterior_sample_dict_init,
  )

  posterior_sample_init = 0.5 * jnp.ones((phi_dim + theta_dim,))
  target_log_prob_fn(posterior_sample_init)

  inner_kernel = tfm.NoUTurnSampler(
      target_log_prob_fn=target_log_prob_fn,
      step_size=config.mcmc_step_size,
  )

  block_bijectors = [tfb.Sigmoid(), tfb.Identity(), tfb.Softplus()]
  block_sizes = [phi_dim, 1, 1]
  kernel_bijector = tfb.Blockwise(
      bijectors=block_bijectors, block_sizes=block_sizes)

  kernel = tfp.mcmc.TransformedTransitionKernel(
      inner_kernel=inner_kernel, bijector=kernel_bijector)

  times_data['start_mcmc_stg_1'] = time.perf_counter()
  posterior_sample = tfm.sample_chain(
      num_results=config.num_samples,
      num_burnin_steps=config.num_burnin_steps,
      kernel=kernel,
      current_state=posterior_sample_init,
      trace_fn=None,
      seed=next(prng_seq),
  )

  posterior_sample_dict = {}
  posterior_sample_dict['phi'], posterior_sample_dict['theta'] = jnp.split(
      posterior_sample, [phi_dim], axis=-1)

  logging.info(
      f"posterior means phi {posterior_sample_dict['phi'].mean(axis=0)}")

  times_data['end_mcmc_stg_1'] = time.perf_counter()

  ### Sample Second Stage ###
  if (smi_eta['modules'][0][1] != 1. if smi_eta else False):
    posterior_sample_dict['theta_aux'] = posterior_sample_dict['theta']
    del posterior_sample_dict['theta']

    logging.info(
        f"posterior means theta_aux {posterior_sample_dict['theta_aux'].mean(axis=0)}"
    )

    logging.info("\t sampling stage 2...")

    def sample_theta(
        phi: Array,
        theta_init: Array,
        num_burnin_steps: int,
        prng_key: PRNGKey,
    ):
      target_log_prob_fn_stage2 = lambda state: log_prob_fn(
          batch=train_ds,
          model_params=jnp.concatenate([phi, state], axis=-1),
          smi_eta_modules=jnp.array([[1., 1.]]),
          model_params_init=posterior_sample_dict_init,
      )
      inner_kernel = tfm.NoUTurnSampler(
          target_log_prob_fn=target_log_prob_fn_stage2,
          step_size=config.mcmc_step_size,
      )
      block_bijectors = [
          tfb.Identity(),
          tfb.Softplus(),
      ]
      block_sizes = [
          1,
          1,
      ]
      kernel_bijectors = [
          tfb.Blockwise(bijectors=block_bijectors, block_sizes=block_sizes)
      ]

      kernel = tfp.mcmc.TransformedTransitionKernel(
          inner_kernel=inner_kernel, bijector=kernel_bijectors)

      posterior_sample = tfm.sample_chain(
          num_results=1,
          num_burnin_steps=num_burnin_steps,
          kernel=kernel,
          current_state=theta_init,
          trace_fn=None,
          seed=prng_key,
      )

      return posterior_sample

    # Example of one sample of theta
    # sample_theta(
    #     phi=posterior_sample_dict['phi'][0, :],
    #     theta_init=posterior_sample_dict['theta_aux'][0, :],
    #     num_burnin_steps=100,
    #     prng_key=next(prng_seq),
    # )

    sample_theta_vmap = jax.vmap(lambda phi, theta_init, prng_key: sample_theta(
        phi=phi,
        theta_init=theta_init,
        num_burnin_steps=config.num_samples_subchain - 1,
        prng_key=prng_key,
    ))

    times_data['start_mcmc_stg_2'] = time.perf_counter()
    theta_vmap = sample_theta_vmap(
        posterior_sample_dict['phi'],
        posterior_sample_dict['theta_aux'],
        jax.random.split(next(prng_seq), config.num_samples),
    )

    posterior_sample_dict['theta'] = theta_vmap.squeeze(1)

  logging.info(
      f"posterior means theta {posterior_sample_dict['theta'].mean(axis=0)}")

  times_data['end_mcmc_stg_2'] = time.perf_counter()

  times_data['end_sampling'] = time.perf_counter()

  logging.info("Sampling times:")
  logging.info(
      f"\t Total: {times_data['end_sampling']-times_data['start_sampling']}")
  logging.info(
      f"\t Stg 1: {times_data['end_mcmc_stg_1']-times_data['start_mcmc_stg_1']}"
  )
  logging.info(
      f"\t Stg 2: {times_data['end_mcmc_stg_2']-times_data['start_mcmc_stg_2']}"
  )

  logging.info("Plotting results...")
  ### Plot SMI samples ###
  plot.posterior_samples(
      posterior_sample_dict=posterior_sample_dict,
      summary_writer=summary_writer,
      step=0,
      eta=smi_eta['modules'][0][1],
      workdir_png=workdir,
  )

  # j = 2
  # fig, axs = plt.subplots(2, 1)
  # axs[0].hist(posterior_sample_dict['phi'][:, j], 30)
  # axs[0].axvline(
  #     x=(train_ds['Z'] / train_ds['N'])[j], color='red', linestyle='dashed')
  # axs[0].set_xlim(-0.01, 1.01)
  # axs[1].plot(posterior_sample_dict['phi'][:, j])

  # j = 13
  # fig, axs = plt.subplots(2, 1)
  # axs[0].hist(states[:, j], 30)
  # axs[1].plot(states[:, j])

  return posterior_sample_dict


# # For debugging
# config = get_config()
# eta = 0.001
# config.smi_eta = {'modules': [[1.0, eta]]}
# workdir = pathlib.Path.home() / f'smi/output/epidemiology/mcmc/eta_{eta:.3f}'
# sample_and_evaluate(config, workdir)
