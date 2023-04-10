"""Plot methods for the epidemiology model."""

from typing import Dict, Any, Optional, Tuple
import warnings

import pathlib

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import arviz as az
from arviz import InferenceData

from modularbayes import plot_to_image, normalize_images

from flax.metrics import tensorboard

from log_prob_fun import ModelParams


def arviz_from_samples(
    dataset: Dict[str, Any],
    model_params: ModelParams,
) -> InferenceData:
  """Converts a samples to an ArviZ InferenceData object.

  Returns:
    ArviZ InferenceData object.
  """
  num_groups = len(dataset['num_obs_groups'])
  assert model_params.sigma.ndim == 3 and (
      model_params.sigma.shape[0] < model_params.sigma.shape[1]), (
          "Arrays in model_params_global" +
          "are expected to have shapes: (num_chains, num_samples, ...)")

  samples_dict = model_params._asdict()
  assert samples_dict['beta'].shape[-1] == num_groups
  assert samples_dict['sigma'].shape[-1] == num_groups

  coords_ = {
      "groups": range(num_groups),
  }
  dims_ = {
      "beta": ["groups"],
      "sigma": ["groups"],
  }

  az_data = az.convert_to_inference_data(
      samples_dict,
      coords=coords_,
      dims=dims_,
  )

  return az_data


def posterior_plots(
    az_data: InferenceData,
    show_sigma_trace: bool,
    show_beta_trace: bool,
    show_tau_trace: bool,
    betasigma_pairplot_groups: Optional[Tuple[int]] = None,
    tausigma_pairplot_groups: Optional[Tuple[int]] = None,
    suffix: str = '',
    step: Optional[int] = 0,
    workdir_png: Optional[str] = None,
    summary_writer: Optional[tensorboard.SummaryWriter] = None,
) -> None:
  model_name = 'random_effects'
  if show_sigma_trace:
    param_name_ = "sigma"
    axs = az.plot_trace(
        az_data,
        var_names=[param_name_],
        compact=False,
    )
    max_ = float(az_data.posterior[param_name_].max())
    for axs_i in axs:
      axs_i[0].set_xlim([0, max_])
    plt.tight_layout()
    if workdir_png:
      plot_name = f"{model_name}_{param_name_}_trace"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(None)

    if summary_writer:
      plot_name = f"{model_name}_{param_name_}_trace"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if show_beta_trace:
    param_name_ = "beta"
    axs = az.plot_trace(
        az_data,
        var_names=[param_name_],
        compact=False,
    )
    max_ = float(az_data.posterior[param_name_].max())
    for axs_i in axs:
      axs_i[0].set_xlim([0, max_])
    plt.tight_layout()
    if workdir_png:
      plot_name = f"{model_name}_{param_name_}_trace"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(None)

    if summary_writer:
      plot_name = f"{model_name}_{param_name_}_trace"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if show_tau_trace:
    param_name_ = "tau"
    axs = az.plot_trace(
        az_data,
        var_names=[param_name_],
        compact=False,
    )
    max_ = float(az_data.posterior[param_name_].max())
    for axs_i in axs:
      axs_i[0].set_xlim([0, max_])
    plt.tight_layout()
    if workdir_png:
      plot_name = f"{model_name}_{param_name_}_trace"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(None)

    if summary_writer:
      plot_name = f"{model_name}_{param_name_}_trace"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if betasigma_pairplot_groups is not None:
    images = []
    n_samples = np.prod(az_data.posterior['tau'].shape[:2])
    for group_i in betasigma_pairplot_groups:
      axs = az.plot_pair(
          az_data,
          var_names=['beta', 'sigma'],
          coords={"groups": group_i},
          kind=["scatter"],
          marginals=True,
          scatter_kwargs={"alpha": np.clip(100 / n_samples, 0.01, 1.)},
          marginal_kwargs={'fill_kwargs': {
              'alpha': 0.10
          }},
      )
      axs[1, 0].set_xlim([-3, 12])
      axs[1, 0].set_ylim([0, 20])
      plt.suptitle(f"group : {group_i}")
      images.append(plot_to_image(None))
    if summary_writer:
      plot_name = f"{model_name}_beta_sigma_pairplot"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )

  if tausigma_pairplot_groups is not None:
    images = []
    n_samples = np.prod(az_data.posterior['tau'].shape[:2])
    for group_i in tausigma_pairplot_groups:
      axs = az.plot_pair(
          az_data,
          var_names=['tau', 'sigma'],
          coords={"groups": group_i},
          kind=["scatter"],
          marginals=True,
          scatter_kwargs={"alpha": np.clip(100 / n_samples, 0.01, 1.)},
          marginal_kwargs={'fill_kwargs': {
              'alpha': 0.10
          }},
      )
      axs[1, 0].set_xlim([0, 4])
      axs[1, 0].set_ylim([0, 20])
      plt.suptitle(f"group : {group_i}")
      images.append(plot_to_image(None))
    if summary_writer:
      plot_name = f"{model_name}_tau_sigma_pairplot"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=normalize_images(images),
          step=step,
          max_outputs=len(images),
      )
