# Scalable Semi-Modular Inference with Variational Meta-Posteriors

<!-- badges: start -->
[![License:
MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/chriscarmona/modularbayes/blob/main/LICENSE)
[![DOI:10.48550/arXiv.2204.00296](https://zenodo.org/badge/DOI/10.48550/arXiv.2204.00296.svg)](https://doi.org/10.48550/arXiv.2204.00296)
[![arXiv](https://img.shields.io/badge/arXiv-2003.06804-b31b1b.svg)](https://arxiv.org/abs/2204.00296)
<!-- badges: end -->

This repo contain the implementation of variational methods described in the article *Scalable Semi-Modular Inference with Variational Meta-Posteriors*.

__Abstract:__

The Cut posterior and related Semi-Modular Inference (SMI) are Generalised Bayes methods for Modular Bayesian evidence combination. Analysis is broken up over modular sub-models of the joint posterior distribution. Model- misspecification in multi-modular models can be hard to fix by model elaboration alone and the Cut posterior and SMI offer a way round this. Information entering the analysis from misspecified modules is controlled by an influence parameter η related to the learning rate.

The article contains two substantial new methods. First, we give variational methods for approximating the Cut and SMI posteriors which are adapted to the inferential goals of evidence combination. We parameterise a family of variational posteriors using a Normalizing Flow for accurate approximation and end-to-end training. Secondly, we show that analysis of models with multiple cuts is feasible using a new Variational Meta-Posterior. This approximates a family of SMI posteriors indexed by η using a single set of variational parameters.

## Examples

We provide code to replicate all the experiments from our article in the [examples](https://github.com/chriscarmona/modularbayes/tree/main/examples) folder.
There are main two examples: 1) epidemiological model and 2) random effects model.
By executing the `run.sh` bash script in that folder, one can train all variational posteriors and produce visualizations and summaries (follow [*Installation instructions*](#installation-instructions) and examine `run.sh` before execution).

You can customize the output directories within the script and choose the experiments that you wish to run.
```bash
pip install -Ur examples/requirements.txt
chmod +x examples/run.sh
bash examples/run.sh
```
Results produced during the optimization can be monitored in [Tensorboard](https://www.tensorflow.org/tensorboard):
```bash
WORK_DIR=$HOME/modularbayes-output
tensorboard --logdir=$WORK_DIR
```

## Installation instructions

1. \[Optional] Create a new virtual environment for this project (see [*Create a virtual environment*](#create-a-virtual-environment) below).
2. Install JAX. This may vary according to your CUDA version (See [JAX installation](https://github.com/google/jax#installation)).
3. Install the latest released version of `modularbayes` from [Pypi](https://pypi.org/project/modularbayes/) via:
```bash
pip install -U modularbayes
```
or you can install the latest development version from GitHub:
```bash
pip install git+https://github.com/chriscarmona/modularbayes
```

## Citation

If you find this work relevant for your scientific publication, we encourage you to add the following reference:

```bibtex
@misc{Carmona2022scalable,
    title = {Scalable Semi-Modular Inference with Variational Meta-Posteriors},
    year = {2022},
    author = {Carmona, Chris U. and Nicholls, Geoff K.},
    month = {4},
    url = {http://arxiv.org/abs/2204.00296},
    doi = {10.48550/arXiv.2204.00296},
    arxivId = {2204.00296},
    keywords = {Cut models, Generalized Bayes, Model misspecification, Scalable inference, Variational Bayes}
}

@InProceedings{Carmona2020smi,
  title = {Semi-Modular Inference: enhanced learning in multi-modular models by tempering the influence of components},
  author = {Carmona, Chris U. and Nicholls, Geoff K.},
  booktitle = {Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics, AISTATS 2020},
  year = {2020},
  editor = {Silvia Chiappa and Roberto Calandra},
  volume = {108},
  pages = {4226--4235},
  series = {Proceedings of Machine Learning Research},
  month = {26--28 Aug},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v108/carmona20a/carmona20a.pdf},
  url = {http://proceedings.mlr.press/v108/carmona20a.html},
  arxivId = {2003.06804},
}
```

### Creating a virtual environment

For OSX or Linux, you can use `venv` (see the [venv documentation](https://docs.python.org/3/library/venv.html)).

```bash
rm -rf ~/.virtualenvs/modularbayes
python3 -m venv ~/.virtualenvs/modularbayes
source ~/.virtualenvs/modularbayes/bin/activate
pip install -U pip
pip install -U setuptools wheel
```

Feel free to change the folder where the virtual environment is created by replacing `~/.virtualenvs/modularbayes` with a path of your choice in both commands.

