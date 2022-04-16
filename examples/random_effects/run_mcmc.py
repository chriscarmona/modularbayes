"""MCMC sampling for the Epidemiology model."""

import time
from absl import logging

import ml_collections

import numpy as np

import jax
from jax import numpy as jnp

from tensorflow_probability.substrates import jax as tfp

from flax.metrics import tensorboard

import haiku as hk

import matplotlib

import log_prob_fun
import plot

from modularbayes import flatten_dict
from modularbayes._src.typing import Any, Dict, Mapping, Optional, OrderedDict

tfd = tfp.distributions
tfb = tfp.bijectors
tfm = tfp.mcmc

np.set_printoptions(suppress=True, precision=4)

matplotlib.use('agg')

Array = jnp.ndarray
PRNGKey = Array
Batch = Mapping[str, np.ndarray]
ConfigDict = ml_collections.ConfigDict
SmiEta = Mapping[str, np.ndarray]


def get_dataset(
    num_obs_groups: Array,
    loc_groups: Array,
    scale_groups: Array,
    prng_key: PRNGKey,
) -> Dict[str, Array]:
  """Generate random effects data as in Liu 2009."""

  num_groups = len(num_obs_groups)
  assert len(loc_groups) == num_groups
  assert len(scale_groups) == num_groups

  num_obs_groups = jnp.array(num_obs_groups).astype(int)
  loc_groups = jnp.array(loc_groups)
  scale_groups = jnp.array(scale_groups)

  loc_pointwise = jnp.repeat(loc_groups, num_obs_groups)
  scale_pointwise = jnp.repeat(scale_groups, num_obs_groups)

  z = jax.random.normal(key=prng_key, shape=loc_pointwise.shape)

  data = {
      'Y': loc_pointwise + z * scale_pointwise,
      'group': jnp.repeat(jnp.arange(num_groups), num_obs_groups),
      'num_obs_groups': num_obs_groups,
  }

  return data


@jax.jit
def log_prob_fn(
    batch: Batch,
    model_params: Array,
    smi_eta_groups: Optional[Array],
    model_params_init: Mapping[str, Any],
):

  leaves_init, treedef = jax.tree_util.tree_flatten(model_params_init)

  leaves = []
  for i in range(len(leaves_init) - 1):
    param_i, model_params = jnp.split(
        model_params, leaves_init[i].flatten().shape, axis=-1)
    leaves.append(param_i.reshape(leaves_init[i].shape))
  leaves.append(model_params.reshape(leaves_init[-1].shape))

  posterior_sample_dict = jax.tree_util.tree_unflatten(
      treedef=treedef, leaves=leaves)

  smi_eta = {'groups': smi_eta_groups} if smi_eta_groups is not None else None

  log_prob = log_prob_fun.log_prob_joint(
      batch=batch,
      posterior_sample_dict=posterior_sample_dict,
      smi_eta=smi_eta,
  ).squeeze()

  return log_prob


def sample_and_evaluate(config: ConfigDict, workdir: str) -> Mapping[str, Any]:

  # Initialize random keys
  prng_seq = hk.PRNGSequence(config.seed)

  # Full dataset used everytime
  # Small data, no need to batch
  train_ds = get_dataset(
      num_obs_groups=config.num_obs_groups,
      loc_groups=config.loc_groups,
      scale_groups=config.scale_groups,
      prng_key=next(prng_seq),
  )

  if jax.process_index() == 0:
    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(flatten_dict(config))

  smi_eta = config.smi_eta
  if smi_eta is not None:
    smi_eta = {k: jnp.array(v) for k, v in smi_eta.items()}

  posterior_sample_dict_init = OrderedDict()
  posterior_sample_dict_init['sigma'] = 1. * jnp.ones((1, config.num_groups))
  posterior_sample_dict_init['beta'] = 0. * jnp.ones((1, config.num_groups))
  posterior_sample_dict_init['tau'] = 1. * jnp.ones((1, 1))

  ### Sample First Stage ###

  logging.info("\t sampling stage 1...")

  times_data = {}
  times_data['start_sampling'] = time.perf_counter()

  target_log_prob_fn = lambda state: log_prob_fn(
      batch=train_ds,
      model_params=state,
      smi_eta_groups=smi_eta['groups'] if smi_eta is not None else None,
      model_params_init=posterior_sample_dict_init,
  )

  posterior_sample_init = jnp.concatenate([
      posterior_sample_dict_init['sigma'],
      posterior_sample_dict_init['beta'],
      posterior_sample_dict_init['tau'],
  ],
                                          axis=-1)[0, :]
  target_log_prob_fn(posterior_sample_init)

  inner_kernel = tfm.NoUTurnSampler(
      target_log_prob_fn=target_log_prob_fn,
      step_size=config.mcmc_step_size,
  )

  # Bijector for mapping values to parameter domain
  # tau goes to [0,Inf]
  # beta goes to [-Inf,Inf]
  # sigma goes to [0,Inf]

  block_bijectors = [tfb.Softplus(), tfb.Identity(), tfb.Softplus()]
  block_sizes = [config.num_groups, config.num_groups, 1]
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
  (posterior_sample_dict['sigma'], posterior_sample_dict['beta'],
   posterior_sample_dict['tau']) = jnp.split(
       posterior_sample, [config.num_groups, 2 * config.num_groups], axis=-1)

  logging.info("posterior means sigma %s",
               str(posterior_sample_dict['sigma'].mean(axis=0)))

  times_data['end_mcmc_stg_1'] = time.perf_counter()

  ### Sample Second Stage ###
  if smi_eta is not None:
    posterior_sample_dict['beta_aux'] = posterior_sample_dict['beta']
    del posterior_sample_dict['beta']
    posterior_sample_dict['tau_aux'] = posterior_sample_dict['tau']
    del posterior_sample_dict['tau']

    logging.info("\t sampling stage 2...")

    def sample_stage2(
        sigma: Array,
        beta_init: Array,
        tau_init: Array,
        num_burnin_steps: int,
        prng_key: PRNGKey,
    ):
      target_log_prob_fn_stage2 = lambda state: log_prob_fn(
          batch=train_ds,
          model_params=jnp.concatenate([sigma, state], axis=-1),
          smi_eta_groups=jnp.array(1.),
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
          config.num_groups,
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
          current_state=jnp.concatenate([beta_init, tau_init], axis=-1),
          trace_fn=None,
          seed=prng_key,
      )

      return posterior_sample

    # Define function to parallelize sample_stage2
    # TODO: use pmap
    sample_stage2_vmap = jax.vmap(
        lambda sigma, beta_init, tau_init, prng_key: sample_stage2(
            sigma=sigma,
            beta_init=beta_init,
            tau_init=tau_init,
            num_burnin_steps=config.num_samples_subchain - 1,
            prng_key=prng_key,
        ))

    def _sample_stage2_loop(num_devices):
      # Initialize beta and tau
      beta = [posterior_sample_dict['beta_aux'][:num_devices, :]]
      tau = [posterior_sample_dict['tau_aux'][:num_devices, :]]

      # Get sequential samples of beta and tau
      assert (config.num_samples % num_devices) == 0

      for i in range(int(config.num_samples / num_devices)):
        beta_tau_i = sample_stage2_vmap(
            posterior_sample_dict['sigma'][(i * num_devices):((i + 1) *
                                                              num_devices), :],
            beta[-1],
            tau[-1],
            jax.random.split(next(prng_seq), num_devices),
        )
        beta_i, tau_i = jnp.split(
            beta_tau_i.squeeze(1), [config.num_groups], axis=-1)
        beta.append(beta_i)
        tau.append(tau_i)

      beta = jnp.concatenate(beta[1:], axis=0)
      tau = jnp.concatenate(tau[1:], axis=0)

      return beta, tau

    # Sample beta and tau
    times_data['start_mcmc_stg_2'] = time.perf_counter()
    posterior_sample_dict['beta'], posterior_sample_dict[
        'tau'] = _sample_stage2_loop(num_devices=int(config.num_samples / 20))

  logging.info("posterior means theta %s",
               str(posterior_sample_dict['theta'].mean(axis=0)))

  times_data['end_mcmc_stg_2'] = time.perf_counter()

  times_data['end_sampling'] = time.perf_counter()

  logging.info("Sampling times:")
  logging.info("\t Total: %s",
               str(times_data['end_sampling'] - times_data['start_sampling']))
  logging.info(
      "\t Stg 1: %s",
      str(times_data['end_mcmc_stg_1'] - times_data['start_mcmc_stg_1']))
  if smi_eta is not None:
    logging.info(
        "\t Stg 2: %s",
        str(times_data['end_mcmc_stg_2'] - times_data['start_mcmc_stg_2']))

  ### Plot SMI samples ###
  plot.posterior_samples(
      posterior_sample_dict=posterior_sample_dict,
      summary_writer=summary_writer,
      step=0,
      suffix=config.plot_suffix,
      workdir_png=workdir,
  )

  # j = 1
  # fig, axs = plt.subplots(2, 1)
  # axs[0].hist(posterior_sample_dict['beta'][:, j], 30)
  # axs[1].plot(posterior_sample_dict['beta'][:, j])

  return posterior_sample_dict
