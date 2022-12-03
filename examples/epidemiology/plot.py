"""Plot methods for the epidemiology model."""

import warnings

import pathlib

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import seaborn as sns

from modularbayes import colour_fader, plot_to_image, normalize_images

from modularbayes._src.typing import (Any, Array, Mapping, Optional,
                                      SummaryWriter, Tuple)

JointGrid = sns.JointGrid


def plot_phi(
    phi: Array,
    theta: Array,
    eta: Optional[float] = None,
    xlim: Optional[Tuple[int]] = None,
    ylim: Optional[Tuple[int]] = None,
) -> Tuple[Figure, Axes]:
  """Relationship between HPV prevalence vs Cancer Incidence ."""

  n_samples, phi_dim = phi.shape

  loglambda = theta[:, [0]] + theta[:, [1]] * phi

  if xlim is None:
    xlim = (None, None)
  if ylim is None:
    ylim = (None, None)

  fig, ax = plt.subplots(figsize=(6, 4))
  for m in range(phi_dim):
    ax.scatter(
        phi[:, m],
        loglambda[:, m],
        alpha=np.clip(100 / n_samples, 0., 1.),
        label=f'{m+1}')
  ax.set(
      title="HPV/Cancer model" + ("" if eta is None else f"\n eta {eta:.3f}"),
      xlabel='phi',
      ylabel='theta_0 + theta_1 * phi',
      xlim=xlim,
      ylim=ylim,
  )
  leg = ax.legend(title='Group', loc='lower center', ncol=4, fontsize='x-small')
  for lh in leg.legendHandles:
    lh.set_alpha(1)

  return fig, ax


def plot_theta(
    theta: Array,
    eta: Optional[float] = None,
    xlim: Optional[Tuple[int]] = None,
    ylim: Optional[Tuple[int]] = None,
) -> JointGrid:
  """Posterior distribution of theta in the epidemiology model."""

  n_samples, theta_dim = theta.shape

  colour = colour_fader('black', 'gold', (eta if eta is not None else 1.0))

  posterior_samples_df = pd.DataFrame(
      theta, columns=[f"theta_{i}" for i in range(1, theta_dim + 1)])
  # fig, ax = plt.subplots(figsize=(10, 10))
  pars = {
      'alpha': np.clip(100 / n_samples, 0., 1.),
      'colour': colour,
  }

  warnings.simplefilter(action='ignore', category=FutureWarning)
  grid = sns.JointGrid(
      x='theta_1',
      y='theta_2',
      data=posterior_samples_df,
      xlim=xlim,
      ylim=ylim,
      height=5)
  g = grid.plot_joint(
      sns.scatterplot,
      data=posterior_samples_df,
      alpha=pars['alpha'],
      color=pars['colour'])
  sns.kdeplot(
      x=posterior_samples_df['theta_1'],
      ax=g.ax_marg_x,
      shade=True,
      color=pars['colour'],
      legend=False)
  sns.kdeplot(
      y=posterior_samples_df['theta_2'],
      ax=g.ax_marg_y,
      shade=True,
      color=pars['colour'],
      legend=False)
  # Add title
  g.fig.subplots_adjust(top=0.9)
  g.fig.suptitle("HPV/Cancer model" +
                 ("" if eta is None else f"\n eta {eta:.3f}"))

  return grid


def posterior_samples(
    posterior_sample_dict: Mapping[str, Any],
    step: int,
    summary_writer: Optional[SummaryWriter] = None,
    eta: Optional[float] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Visualise samples from the approximate posterior distribution."""

  images = []

  # Plot relation: prevalence (lambda) vs incidence (phi)
  plot_name = "epidemiology_phi"
  plot_name = plot_name + ("" if (eta is None) else f"_eta_{eta:.3f}")
  fig, _ = plot_phi(
      posterior_sample_dict['phi'],
      posterior_sample_dict['theta'],
      eta=1.0 if (eta is None) else float(eta),
      xlim=[0, 0.3],
      ylim=[-2.5, 2],
  )
  if workdir_png:
    fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
  # summary_writer.image(plot_name, plot_to_image(fig), step=step)
  images.append(plot_to_image(fig))

  # Plot relation: theta_2 vs theta_1
  plot_name = "epidemiology_theta"
  plot_name = plot_name + ("" if (eta is None) else f"_eta_{eta:.3f}")
  grid = plot_theta(
      posterior_sample_dict['theta'],
      eta=1.0 if (eta is None) else float(eta),
      xlim=[-3, -1],
      ylim=[5, 35],
  )
  fig = grid.fig
  if workdir_png:
    fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
  # summary_writer.image(plot_name, plot_to_image(fig), step=step)
  images.append(plot_to_image(fig))

  # Log all images
  if summary_writer:
    plot_name = "epidemiology_posterior_samples"
    plot_name = plot_name + ("" if (eta is None) else f"_eta_{eta:.3f}")
    summary_writer.image(
        tag=plot_name,
        image=normalize_images(images),
        step=step,
        max_outputs=len(images),
    )
