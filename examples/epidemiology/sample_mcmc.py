"""MCMC sampling for the model."""

import os
import pathlib
import time

from absl import logging

import numpy as np

import arviz as az

import jax
from jax import numpy as jnp

import haiku as hk
import distrax

import blackjax
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.nuts import NUTSInfo

from tensorflow_probability.substrates import jax as tfp

from objax.jaxboard import SummaryWriter, Summary

from modularbayes._src.typing import (
    Any,
    Array,
    ArrayTree,
    Batch,
    Callable,
    ConfigDict,
    Dict,
    Mapping,
    Optional,
    PRNGKey,
    Tuple,
)

import plot
from train_flow import load_data
import log_prob_fun
from log_prob_fun import ModelParams, ModelParamsCut, ModelParamsNoCut, SmiEta
from flows import (
    bijector_domain_nocut,
    bijector_domain_cut,
    split_flow_nocut,
    split_flow_cut,
    concat_flow_nocut,
    concat_flow_cut,
)

tfb = tfp.bijectors
kernels = tfp.math.psd_kernels

# Set high precision for matrix multiplication in jax
jax.config.update("jax_default_matmul_precision", "float32")

np.set_printoptions(suppress=True, precision=4)


def call_warmup(
    prng_key: PRNGKey,
    logdensity_fn: Callable,
    model_params: ArrayTree,
    num_steps: int,
) -> Tuple:
  warmup = blackjax.window_adaptation(
      algorithm=blackjax.nuts,
      logdensity_fn=logdensity_fn,
  )
  (initial_states, hmc_params), _ = warmup.run(
      rng_key=prng_key,
      position=model_params,
      num_steps=num_steps,
  )
  return initial_states, hmc_params


def inference_loop_stg1(
    prng_key: PRNGKey,
    initial_states: HMCState,
    hmc_params: Dict[str, Array],
    logdensity_fn: Callable,
    num_samples: int,
    num_chains: int,
) -> Tuple[HMCState, NUTSInfo]:

  def one_step(states, rng_keys):
    kernel_fn_multichain = jax.vmap(
        lambda state_, hmc_param_, key_nuts_: blackjax.nuts(
            logdensity_fn=logdensity_fn,
            step_size=hmc_param_["step_size"],
            inverse_mass_matrix=hmc_param_["inverse_mass_matrix"],
        ).step(
            rng_key=key_nuts_,
            state=state_,
        ))
    states_new, infos_new = kernel_fn_multichain(
        states,
        hmc_params,
        rng_keys,
    )
    return states_new, (states_new, infos_new)

  keys = jax.random.split(prng_key, num_samples * num_chains).reshape(
      (num_samples, num_chains, 2))

  # one_step(initial_states, keys[0])

  _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

  return states, infos


def inference_loop_stg2(
    prng_key: PRNGKey,
    initial_states: HMCState,
    hmc_params: Dict[str, Array],
    logdensity_fn_conditional: Callable,
    conditioner_logprob: ModelParamsNoCut,
    num_samples_stg1: int,
    num_samples_stg2: int,
    num_chains: int,
):
  # We only need to keep the last sample of the subchains
  def one_step(states, rng_keys):
    kernel_fn_multichain = jax.vmap(
        lambda state, cond, hmc_param, key_nuts_: blackjax.nuts(
            logdensity_fn=lambda x: logdensity_fn_conditional(
                model_params_sampled=x,
                model_params_cond=cond,
            ),
            step_size=hmc_param["step_size"],
            inverse_mass_matrix=hmc_param["inverse_mass_matrix"],
        ).step(
            rng_key=key_nuts_,
            state=state,
        ))
    kernel_fn_multicond_multichain = jax.vmap(
        lambda states_, conds_, keys_nuts_: kernel_fn_multichain(
            states_,
            conds_,
            hmc_params,
            keys_nuts_,
        ))
    states_new, _ = kernel_fn_multicond_multichain(
        states,
        conditioner_logprob,
        rng_keys,
    )
    return states_new, None

  keys = jax.random.split(
      prng_key, num_samples_stg2 * num_samples_stg1 * num_chains).reshape(
          (num_samples_stg2, num_samples_stg1, num_chains, 2))

  # one_step(initial_states, keys[0])

  last_state, _ = jax.lax.scan(one_step, initial_states, keys)

  return last_state


def init_param_fn_stg1(
    prng_key: PRNGKey,
    num_groups: int,
) -> ModelParams:
  """Get dictionary with parametes to initialize MCMC.

  This function produce unbounded values, i.e. before bijectors to map into the
  domain of the model parameters.
  """
  prng_seq = hk.PRNGSequence(prng_key)
  # Dictionary with shapes of model parameters in stage 1
  samples_shapes = {
      "phi": (num_groups,),
      "theta0": (1,),
      "theta1": (1,),
  }
  # Get a sample for all parameters
  samples_ = jax.tree_map(
      lambda shape_i: distrax.Normal(0.0, 1.0).sample(
          seed=next(prng_seq), sample_shape=shape_i),
      tree=samples_shapes,
      is_leaf=lambda x: isinstance(x, Tuple),
  )
  # Define the namedtuple
  model_params_stg1 = ModelParams(**samples_)
  return model_params_stg1


def transform_model_params(
    model_params_unb: ModelParams) -> Tuple[ModelParams, Array]:
  """Apply transformations to map into model parameters domain."""

  phi_dim = model_params_unb.phi.shape[-1]
  theta_dim = 2

  # Transform no-cut parameters
  concat_params_nocut_unb = concat_flow_nocut(
      ModelParamsNoCut(
          **{k: model_params_unb._asdict()[k]
             for k in ModelParamsNoCut._fields}))
  bij_nocut = bijector_domain_nocut()
  concat_params_nocut, log_det_jacob_nocut = bij_nocut.forward_and_log_det(
      concat_params_nocut_unb)
  model_params_nocut = split_flow_nocut(
      concat_params=concat_params_nocut, phi_dim=phi_dim)

  # Transform cut parameters
  concat_params_cut_unb = concat_flow_cut(
      ModelParamsCut(
          **{k: model_params_unb._asdict()[k] for k in ModelParamsCut._fields}))
  bij_cut = bijector_domain_cut()
  concat_params_cut, log_det_jacob_cut = bij_cut.forward_and_log_det(
      concat_params_cut_unb)
  model_params_cut = split_flow_cut(
      concat_params=concat_params_cut, theta_dim=theta_dim)
  # Join model parameters
  model_params = ModelParams(**{
      **model_params_nocut._asdict(),
      **model_params_cut._asdict()
  })

  # Total log determinant of the Jacobian
  log_det_jacob = log_det_jacob_nocut + log_det_jacob_cut

  return model_params, log_det_jacob


def logprob_joint_unb(
    batch: Batch,
    model_params_unb: ModelParams,
    prior_hparams: Optional[Dict[str, float]] = None,
    smi_eta: Optional[SmiEta] = None,
):
  """Joint log probability of the model taking unbounded input parameters."""
  (model_params, log_det_jacob) = transform_model_params(model_params_unb)
  log_prob = log_prob_fun.logprob_joint(
      batch=batch,
      model_params=model_params,
      prior_hparams=prior_hparams,
      smi_eta=smi_eta,
  )
  return log_prob + log_det_jacob


def sample_and_evaluate(config: ConfigDict, workdir: str) -> Mapping[str, Any]:
  """Sample and evaluate the random effects model."""

  # output directory
  workdir = workdir.rstrip("/") + "/"
  pathlib.Path(workdir).mkdir(parents=True, exist_ok=True)

  # Initialize random keys
  prng_seq = hk.PRNGSequence(config.seed)

  # Load and process data
  train_ds = load_data()

  samples_path_stg1 = workdir + "mcmc_samples_stg1_unb_az.nc"
  samples_path = workdir + "mcmc_samples_az.nc"

  # Set eta for modules
  # In general, it would be possible to modulate the influence of both modules
  # for now, we only focus on the influence of the cancer module
  smi_eta = SmiEta(cancer=config.smi_eta_cancer)

  tensorboard = SummaryWriter(workdir)
  if os.path.exists(samples_path):
    logging.info("\t Loading final samples from: %s", samples_path)
    az_data = az.from_netcdf(samples_path)
  else:
    times_data = {}
    times_data["start_sampling"] = time.perf_counter()

    ### Sample First Stage ###
    if os.path.exists(samples_path_stg1):
      logging.info("\t Loading samples for stage 1 from: %s", samples_path_stg1)
      az_data_stg1_unb_az = az.from_netcdf(samples_path_stg1)
    else:
      logging.info("\t Stage 1...")

      # Define target logdensity function
      @jax.jit
      def logdensity_fn_stg1(model_params):
        logprob_ = logprob_joint_unb(
            batch=train_ds,
            model_params_unb=model_params,
            prior_hparams=config.prior_hparams,
            smi_eta=smi_eta,
        )
        return logprob_

      # initial positions of model parameters
      # (vmap to produce one for each MCMC chains)
      model_params_stg1_unb_init = jax.vmap(lambda prng_key: init_param_fn_stg1(
          prng_key=prng_key,
          num_groups=train_ds["Z"].shape[0],
      ))(
          jax.random.split(next(prng_seq), config.num_chains))

      # Tune HMC parameters automatically
      logging.info("\t tuning HMC parameters stg1...")
      initial_states_stg1, hmc_params_stg1 = jax.vmap(
          lambda prng_key, model_params: call_warmup(
              prng_key=prng_key,
              logdensity_fn=logdensity_fn_stg1,
              model_params=model_params,
              num_steps=config.num_steps_call_warmup,
          ))(
              jax.random.split(next(prng_seq), config.num_chains),
              model_params_stg1_unb_init,
          )

      # Sampling loop stage 1
      logging.info("\t sampling stage 1...")
      states_stg1, _ = inference_loop_stg1(
          prng_key=next(prng_seq),
          initial_states=initial_states_stg1,
          hmc_params=hmc_params_stg1,
          logdensity_fn=logdensity_fn_stg1,
          num_samples=config.num_samples,
          num_chains=config.num_chains,
      )

      # Save samples from stage 1
      # swap position axes to have shape (num_chains, num_samples, ...)
      model_params_stg1_unb_samples = jax.tree_map(lambda x: x.swapaxes(0, 1),
                                                   states_stg1.position)
      # Create InferenceData object
      az_data_stg1_unb_az = plot.arviz_from_samples(
          dataset=train_ds,
          model_params=model_params_stg1_unb_samples,
      )

      # Save InferenceData object from stage 1
      az_data_stg1_unb_az.to_netcdf(samples_path_stg1)

      logging.info("\t\t Posterior means no-cut parameters (before transform):")
      for k in az_data_stg1_unb_az.posterior:
        logging.info(
            "\t\t %s: \n %s \n", k,
            str(jnp.array(az_data_stg1_unb_az.posterior[k]).mean(axis=[0, 1])))

      times_data["end_mcmc_stg_1"] = time.perf_counter()

    summary = Summary()
    logging.info("Plotting stage 1 samples...")
    plot.posterior_plots(
        az_data=az_data_stg1_unb_az,
        show_phi_trace=True,
        eta=config.smi_eta_cancer,
        suffix=f"_eta_cancer_{float(config.smi_eta_cancer):.3f}",
        workdir_png=workdir,
        summary=summary,
    )
    # Write to tensorboard
    tensorboard.write(summary=summary, step=1)
    logging.info("...done!")

    ### Sample Second Stage ###
    logging.info("\t Stage 2...")

    # Extract Cut parameters from stage 1 samples
    model_params_condstg2_unb_samples = ModelParamsNoCut(
        **{
            k: jnp.array(az_data_stg1_unb_az.posterior[k])
            for k in ModelParamsNoCut._fields
        })

    # Define target logdensity function
    @jax.jit
    def logdensity_fn_stg2(model_params_sampled, model_params_cond):
      model_params_unb = ModelParams(**{
          **model_params_sampled._asdict(),
          **model_params_cond._asdict(),
      })
      logprob_ = logprob_joint_unb(
          batch=train_ds,
          model_params_unb=model_params_unb,
          prior_hparams=config.prior_hparams,
          smi_eta=None,
      )
      return logprob_

    # Tune HMC parameters automatically
    logging.info("\t tuning HMC parameters stg2...")

    # We tune the HMC for one sample in stage 1
    # tune HMC parameters, vmap across chains, using one sample from stage 1
    _, hmc_params_stg2 = jax.vmap(lambda key, param, cond: call_warmup(
        prng_key=key,
        logdensity_fn=lambda param_: logdensity_fn_stg2(
            model_params_sampled=param_,
            model_params_cond=cond,
        ),
        model_params=param,
        num_steps=config.num_steps_call_warmup,
    ))(
        jax.random.split(next(prng_seq), config.num_chains),
        ModelParamsCut(
            **{
                k: jnp.array(az_data_stg1_unb_az.posterior[k])[:, 0, ...]
                for k in ModelParamsCut._fields
            }),
        jax.tree_map(lambda x: x[:, 0, ...], model_params_condstg2_unb_samples),
    )

    # The number of samples is large and often it does not fit into GPU memory
    # we split the sampling of stage 2 into chunks
    assert config.num_samples % config.num_samples_perchunk_stg2 == 0
    num_chunks_stg2 = config.num_samples // config.num_samples_perchunk_stg2

    # Function to initialize stage 2
    # we use the tuned HMC parameters from above
    # Note: vmap is first applied to the chains, then to samples from
    #   conditioner this requires swap axes 0 and 1 in a few places
    init_fn_multichain = jax.vmap(lambda param, cond, hmc_param: blackjax.nuts(
        logdensity_fn=lambda param_: logdensity_fn_stg2(
            model_params_sampled=param_,
            model_params_cond=cond,
        ),
        step_size=hmc_param["step_size"],
        inverse_mass_matrix=hmc_param["inverse_mass_matrix"],
    ).init(position=param))
    init_fn_multicond_multichain = jax.vmap(
        lambda param_, cond_: init_fn_multichain(
            param=param_,
            cond=cond_,
            hmc_param=hmc_params_stg2,
        ))

    # The initial position for theta in the first chunk is the location
    # of theta_aux from stage 1
    initial_position_i = ModelParamsCut(
        **{
            k:
                jnp.array(az_data_stg1_unb_az.posterior[k])
                [:, :config.num_samples_perchunk_stg2, ...].swapaxes(0, 1)
            for k in ModelParamsCut._fields
        })

    logging.info("\t sampling stage 2...")
    samples_chunks = []
    for i in range(num_chunks_stg2):
      # Take a chunk of samples from the conditional (no-cut) parameters
      cond_i = jax.tree_map(
          lambda x: x[
              :,
              (i * config.num_samples_perchunk_stg2):
              ((i + 1) * config.num_samples_perchunk_stg2),
              ...,
          ].swapaxes(0, 1),
          model_params_condstg2_unb_samples,
      )
      # Get value to initialize sampling in this chunk
      initial_state_i = init_fn_multicond_multichain(initial_position_i, cond_i)
      # Sampling loop for this chunk of stage 2
      states_stg2_i = inference_loop_stg2(
          prng_key=next(prng_seq),
          initial_states=initial_state_i,
          hmc_params=hmc_params_stg2,
          logdensity_fn_conditional=logdensity_fn_stg2,
          conditioner_logprob=cond_i,
          num_samples_stg1=config.num_samples_perchunk_stg2,
          num_samples_stg2=config.num_samples_subchain_stg2,
          num_chains=config.num_chains,
      )
      samples_chunks.append(states_stg2_i.position)
      # Subsequent chunks initialise in last position of the previous chunk
      initial_position_i = states_stg2_i.position

    times_data["end_mcmc_stg_2"] = time.perf_counter()

    # Concatenate samples from each chunk, across samples dimension
    model_params_cut_unb_samples = (
        jax.tree_map(  # pylint: disable=no-value-for-parameter
            lambda *x: jnp.concatenate(x, axis=0), *samples_chunks))
    # swap axes to have shape (num_chains, num_samples, ...)
    model_params_cut_unb_samples = jax.tree_map(lambda x: x.swapaxes(0, 1),
                                                model_params_cut_unb_samples)

    # Transform unbounded parameters to model parameters
    model_params_stg2_samples, _ = jax.vmap(jax.vmap(transform_model_params))(
        ModelParams(
            **{
                **model_params_condstg2_unb_samples._asdict(),
                **model_params_cut_unb_samples._asdict(),
            }))

    # Create InferenceData object
    az_data = plot.arviz_from_samples(
        dataset=train_ds,
        model_params=model_params_stg2_samples,
    )
    # Save InferenceData object
    az_data.to_netcdf(samples_path)

    logging.info("\t\t Posterior means:")
    for k in az_data.posterior:
      logging.info("\t\t %s: \n %s \n", k,
                   str(jnp.array(az_data.posterior[k]).mean(axis=[0, 1])))

    times_data["end_sampling"] = time.perf_counter()

    logging.info("Sampling times:")
    logging.info(
        "\t Total: %s",
        str(times_data["end_sampling"] - times_data["start_sampling"]),
    )
    if ("start_mcmc_stg_1" in times_data) and ("end_mcmc_stg_1" in times_data):
      logging.info(
          "\t Stg 1: %s",
          str(times_data["end_mcmc_stg_1"] - times_data["start_mcmc_stg_1"]),
      )
    if ("start_mcmc_stg_2" in times_data) and ("end_mcmc_stg_2" in times_data):
      logging.info(
          "\t Stg 2: %s",
          str(times_data["end_mcmc_stg_2"] - times_data["start_mcmc_stg_2"]),
      )

  summary = Summary()
  logging.info("Plotting results...")
  plot.posterior_plots(
      az_data=az_data,
      show_phi_trace=True,
      show_theta_trace=True,
      show_loglinear_scatter=True,
      show_theta_pairplot=True,
      eta=config.smi_eta_cancer,
      suffix=f"_eta_cancer_{float(config.smi_eta_cancer):.3f}",
      workdir_png=workdir,
      summary=summary,
  )
  # Write to tensorboard
  tensorboard.write(summary=summary, step=2)
  tensorboard.close()
  logging.info("...done!")
