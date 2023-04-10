"""Plot methods for the epidemiology model."""

from typing import Dict, Any, Optional, Tuple

import pathlib

import numpy as np

from matplotlib import pyplot as plt

import arviz as az
from arviz import InferenceData

import jax
from jax import numpy as jnp
import haiku as hk

from flax.metrics import tensorboard

from modularbayes import colour_fader, plot_to_image
from modularbayes._src.typing import Array

from log_prob_fun import ModelParams, SmiEta


def arviz_from_samples(
    dataset: Dict[str, Any],
    model_params: ModelParams,
) -> InferenceData:
  """Converts a samples to an ArviZ InferenceData object.

  Returns:
    ArviZ InferenceData object.
  """

  # Number of HPV groupss in the data

  assert model_params.phi.ndim == 3 and (
      model_params.phi.shape[0] < model_params.phi.shape[1]), (
          "Arrays in model_params_global" +
          "are expected to have shapes: (num_chains, num_samples, ...)")

  samples_dict = model_params._asdict()

  num_groups = dataset['Z'].shape[0]
  assert samples_dict['phi'].shape[-1] == num_groups

  coords_ = {
      "groups": range(num_groups),
  }
  dims_ = {
      "phi": ["groups"],
  }

  az_data = az.convert_to_inference_data(
      samples_dict,
      coords=coords_,
      dims=dims_,
  )

  return az_data


def posterior_plots(
    az_data: InferenceData,
    show_phi_trace: bool,
    show_theta_trace: bool,
    show_loglinear_scatter: bool,
    show_theta_pairplot: bool,
    eta: Optional[float] = None,
    suffix: str = '',
    step: Optional[int] = 0,
    workdir_png: Optional[str] = None,
    summary_writer: Optional[tensorboard.SummaryWriter] = None,
) -> None:
  model_name = 'epidemiology'
  if show_phi_trace:
    param_name_ = "phi"
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

  if show_theta_trace:
    axs = az.plot_trace(
        az_data,
        var_names=['theta0', 'theta1'],
        compact=False,
    )
    plt.tight_layout()
    if workdir_png:
      plot_name = f"{model_name}_theta_trace"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(None)

    if summary_writer:
      plot_name = f"{model_name}_theta_trace"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if show_theta_pairplot:
    colour = colour_fader('black', 'gold', (eta if eta is not None else 1.0))
    n_samples = np.prod(az_data.posterior['theta0'].shape[:2])
    axs = az.plot_pair(
        az_data,
        var_names=['theta0', 'theta1'],
        kind=["scatter"],
        marginals=True,
        scatter_kwargs={
            "color": colour,
            "alpha": np.clip(100 / n_samples, 0.01, 1.)
        },
        marginal_kwargs={
            "color": colour,
            'fill_kwargs': {
                'alpha': 0.10
            }
        },
    )
    axs[1, 0].set_xlim([-3, -1])
    axs[1, 0].set_ylim([5, 35])
    if workdir_png:
      plot_name = f"{model_name}_theta_pairplot"
      plot_name += suffix
      plt.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(None)

    if summary_writer:
      plot_name = f"{model_name}_theta_pairplot"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )

  if show_loglinear_scatter:
    loglambda = (
        az_data.posterior['theta0'].values +
        az_data.posterior['theta1'].values * az_data.posterior['phi'].values)
    n_samples = np.prod(loglambda.shape[:-1])

    fig, ax = plt.subplots(figsize=(6, 4))
    for m in range(az_data.posterior['phi'].shape[-1]):
      ax.scatter(
          az_data.posterior['phi'].values[..., m].squeeze(),
          loglambda[..., m].squeeze(),
          alpha=np.clip(100 / n_samples, 0.01, 1.),
          label=f'{m+1}')
    ax.set_xlim([0, 0.3])
    ax.set_ylim([-2.5, 2])
    ax.set_xlabel('phi')
    ax.set_ylabel('theta_0 + theta_1 * phi')
    ax.set_title("HPV/Cancer model" +
                 ("" if eta is None else f"\n eta {eta:.3f}"))
    leg = ax.legend(
        title='Group', loc='lower center', ncol=4, fontsize='x-small')
    for lh in leg.legendHandles:
      lh.set_alpha(1)

    if workdir_png:
      plot_name = f"{model_name}_loglinear_scatter"
      plot_name += suffix
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    image = plot_to_image(fig)

    if summary_writer:
      plot_name = f"{model_name}_loglinear_scatter"
      plot_name += suffix
      summary_writer.image(
          tag=plot_name,
          image=image,
          step=step,
      )


def plot_vmp_map(
    alpha: hk.Params,
    vmpmap_fn: hk.Transformed,
    lambda_init: hk.Params,
    lambda_idx_to_plot: Tuple[int],
    eta_cancer_grid: Array,
):
  """Visualize VMP map."""

  eta_cancer_grid = jnp.linspace(0, 1, 51).round(2)
  smi_etas = SmiEta(
      hpv=jnp.ones(len(eta_cancer_grid)),
      cancer=jnp.array(eta_cancer_grid),
  )
  lambda_grid = vmpmap_fn.apply(
      alpha,
      eta_values=(smi_etas[0] if len(smi_etas) == 1 else jnp.stack(
          smi_etas, axis=-1)),
      lambda_init=lambda_init,
  )
  # All variational parameters, concatenated
  lambda_grid_concat = jnp.concatenate([
      x.reshape(len(eta_cancer_grid), -1)
      for x in jax.tree_util.tree_leaves(lambda_grid)
  ],
                                       axis=-1)

  # Plot variational parameters as a function of eta_Y
  fig, axs = plt.subplots(
      nrows=1,
      ncols=len(lambda_idx_to_plot),
      figsize=(4 * len(lambda_idx_to_plot), 3),
      squeeze=False,
  )
  for i, idx_i in enumerate(lambda_idx_to_plot):
    axs[0, i].plot(eta_cancer_grid, lambda_grid_concat[:, idx_i])
    axs[0, i].plot(eta_cancer_grid, lambda_grid_concat[:, idx_i])
    axs[0, i].set_xlabel('eta')
    axs[0, i].set_ylabel(f'lambda_{idx_i}')
  fig.tight_layout()

  return fig, axs
