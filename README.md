# Scalable Semi-Modular Inference with Variational Meta-Posteriors

<!-- badges: start -->
[![License:
MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/chriscarmona/modularbayes/blob/main/LICENSE)
<!-- badges: end -->

This repo contain the implementation of variational methods described in the article *Scalable Semi-Modular Inference with Variational Meta-Posteriors*.

__Abstract:__

The Cut posterior and related Semi-Modular Inference (SMI) are Generalised Bayes methods for Modular Bayesian evidence combination. Analysis is broken up over modular sub-models of the joint posterior distribution. Model- misspecification in multi-modular models can be hard to fix by model elaboration alone and the Cut posterior and SMI offer a way round this. Information entering the analysis from misspecified modules is controlled by an influence parameter η related to the learning rate.

The article contains two substantial new methods. First, we give variational methods for approximating the Cut and SMI posteriors which are adapted to the inferential goals of evidence combination. We parameterise a family of variational posteriors using a Normalizing Flow for accurate approximation and end-to-end training. Secondly, we show that analysis of models with multiple cuts is feasible using a new Variational Meta-Posterior. This approximates a family of SMI posteriors indexed by η using a single set of variational parameters.

## Examples

We include code to replicate all the examples from our article. By executing the `run.sh` bash script one can train all variational posteriors and produce visualizations and summaries (follow [*Installation instructions*](#create-a-virtual-environment) before running the script).

```bash
# Assume we are located at the SMI directory
SMI_DIR=$PWD

# Directory to save all outputs
WORK_DIR=$HOME/smi/output

./examples/run.sh
```

We recommend to monitor training via tensorboard

```bash
tensorboard --logdir=$WORK_DIR
```

## Installation instructions

1. \[Optional] Create a new virtual environment for this project (see [*Create a virtual environment*](#create-a-virtual-environment) below).
2. Install JAX. This may vary according to your CUDA version (See [JAX installation](https://github.com/google/jax#installation)).
3. Clone this repository locally
```bash
git clone https://github.com/chriscarmona/modularbayes
```
4. Install the `modularbayes` module
```bash
pip install -e ./modularbayes
```


## Citation

If you find this work relevant for your scientific publication, we encourage you to add the following reference:

```bibtex
@misc{Carmona2022scalable,
  author = {Carmona, Chris U. and Nicholls, Geoff K.},
  title = {Scalable Semi-Modular Inference with Variational Meta-Posteriors},
  url = {https://www.github.com/chriscarmona/modularbayes},
  year = {2022}
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

Create `modularbayes` virtual environment
```bash
python3 -m venv ~/.virtualenvs/modularbayes
```
Activate the virtual environment
```bash
source ~/.virtualenvs/modularbayes/bin/activate
```
Upgrade pip
```bash
pip install -U pip
```

Feel free to change the folder where the virtual environment is created by replacing `~/.virtualenvs/modularbayes` with a path of your choice in both commands.

