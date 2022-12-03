"""Plot methods for the random effects model."""

import pathlib
import warnings

import math
import numpy as np
import pandas as pd

import seaborn as sns
from seaborn import JointGrid

from modularbayes import plot_to_image, normalize_images
from modularbayes._src.typing import (Any, Array, Dict, Optional, SummaryWriter,
                                      Tuple)


def plot_beta_sigma(
    beta: Array,
    sigma: Array,
    group: int = 1,
    suffix: Optional[str] = None,
    xlim: Optional[Tuple[int]] = None,
    ylim: Optional[Tuple[int]] = None,
) -> JointGrid:

  num_samples, num_groups = sigma.shape

  posterior_samples_df = pd.DataFrame(
      np.concatenate([sigma, beta], axis=-1),
      columns=[f"sigma_{i}" for i in range(1, num_groups + 1)] +
      [f"beta_{i}" for i in range(1, num_groups + 1)],
  )
  # fig, ax = plt.subplots(figsize=(10, 10))
  pars = {'alpha': np.clip(500 / num_samples, 0., 1.), 'colour': 'blue'}

  warnings.simplefilter(action='ignore', category=FutureWarning)
  grid = sns.JointGrid(
      x=f'beta_{group}',
      y=f'sigma_{group}',
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
      x=posterior_samples_df[f'beta_{group}'],
      ax=g.ax_marg_x,
      shade=True,
      color=pars['colour'],
      legend=False)
  sns.kdeplot(
      y=posterior_samples_df[f'sigma_{group}'],
      ax=g.ax_marg_y,
      shade=True,
      color=pars['colour'],
      legend=False)
  # Add title
  g.fig.subplots_adjust(top=0.9)
  g.fig.suptitle("Random effects model" +
                 ('' if suffix is None else f" ({suffix})"))

  return grid


def plot_tau_sigma(
    tau: Array,
    sigma: Array,
    group: int = 1,
    suffix: Optional[str] = None,
    xlim: Optional[Tuple[int]] = None,
    ylim: Optional[Tuple[int]] = None,
) -> JointGrid:

  num_samples, num_groups = sigma.shape

  posterior_samples_df = pd.DataFrame(
      np.concatenate([sigma, tau], axis=-1),
      columns=[f"sigma_{i}" for i in range(1, num_groups + 1)] + ['tau'],
  )
  # fig, ax = plt.subplots(figsize=(10, 10))
  pars = {'alpha': np.clip(500 / num_samples, 0., 1.), 'colour': 'blue'}

  warnings.simplefilter(action='ignore', category=FutureWarning)
  grid = sns.JointGrid(
      x='tau',
      y=f'sigma_{group}',
      data=posterior_samples_df,
      xlim=xlim,
      ylim=ylim,
      height=5,
  )
  g = grid.plot_joint(
      sns.scatterplot,
      data=posterior_samples_df,
      alpha=pars['alpha'],
      color=pars['colour'])
  sns.kdeplot(
      posterior_samples_df['tau'],
      ax=g.ax_marg_x,
      shade=True,
      color=pars['colour'],
      legend=False)
  sns.kdeplot(
      posterior_samples_df[f'sigma_{group}'],
      ax=g.ax_marg_y,
      shade=True,
      color=pars['colour'],
      legend=False,
      vertical=True)
  # Add title
  g.fig.subplots_adjust(top=0.9)
  g.fig.suptitle("Random effects model" +
                 ('' if suffix is None else f" ({suffix})"))

  return grid


def posterior_samples(
    posterior_sample_dict: Dict[str, Any],
    step: int,
    summary_writer: Optional[SummaryWriter] = None,
    suffix: Optional[str] = None,
    workdir_png: Optional[str] = None,
) -> None:
  """Plot posteriors samples."""

  images = []

  # Plot relation: sigma vs beta
  for group in [1, 2, 3]:
    plot_name = f"rnd_eff_beta_sigma_group_{group}"
    plot_name = plot_name + ("" if (suffix is None) else ("_" + suffix))
    grid = plot_beta_sigma(
        beta=posterior_sample_dict['beta'],
        sigma=posterior_sample_dict['sigma'],
        group=group,
        suffix=suffix,
        xlim=(-3, 12),
        ylim=(0, 20),
    )
    fig = grid.fig
    if workdir_png:
      fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
    # summary_writer.image(plot_name, images[-1], step=step)
    images.append(plot_to_image(fig))

  # Plot relation: sigma vs tau, group 1
  plot_name = "rnd_eff_sigma_tau_group_1"
  plot_name = plot_name + ("" if (suffix is None) else ("_" + suffix))
  grid = plot_tau_sigma(
      tau=posterior_sample_dict['tau'],
      sigma=posterior_sample_dict['sigma'],
      group=1,
      suffix=suffix,
      xlim=(0, 4),
      ylim=(0, 20),
  )
  fig = grid.fig
  if workdir_png:
    fig.savefig(pathlib.Path(workdir_png) / (plot_name + ".png"))
  # summary_writer.image(plot_name, plot_to_image(fig), step=step)
  images.append(plot_to_image(fig))

  # Same height and width in all images
  h, w = max([x.shape[1] for x in images]), max([x.shape[2] for x in images])
  for i, img in enumerate(images):
    h_pad = (h - img.shape[1]) / 2
    h_pad = math.floor(h_pad), math.ceil(h_pad)
    w_pad = (w - img.shape[2]) / 2
    w_pad = math.floor(w_pad), math.ceil(w_pad)
    images[i] = np.pad(img, ((0, 0), h_pad, w_pad, (0, 0)), mode='constant')

  # Log all images
  if summary_writer:
    plot_name = "rnd_eff_posterior_samples"
    plot_name = plot_name + ("" if (suffix is None) else ("_" + suffix))
    summary_writer.image(
        tag=plot_name,
        image=normalize_images(images),
        step=step,
        max_outputs=len(images),
    )
