#!/bin/bash

# Assume we are located at the SMI directory
SMI_DIR=$PWD

# Directory to save all outputs
WORK_DIR=$HOME/smi/output

# Create output directory and install missing dependencies
mkdir -p $WORK_DIR
pip install -r $SMI_DIR/requirements/requirements-examples.txt

# Epidemiological data
## One posterior for each eta
eta='(0.001,)'
## MCMC
python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/mcmc.py \
                                               --workdir=$WORK_DIR/epidemiology/mcmc/eta \
                                               --config.iterate_smi_eta=$eta
### Mean field
python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_mf.py \
                                               --workdir=$WORK_DIR/epidemiology/mean_field/eta \
                                               --config.iterate_smi_eta=$eta
### Neural Spline Flow
python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_nsf.py \
                                               --workdir=$WORK_DIR/epidemiology/nsf/eta \
                                               --config.iterate_smi_eta=$eta
## Variational Meta-Posterior
### Mean field, initialized on trained eta=0.100
python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_mf_vmp_map_gp.py \
                                               --workdir=$WORK_DIR/epidemiology/mean_field/vmp_map_gp \
                                               --config.state_flow_init_path=$WORK_DIR/epidemiology/mean_field/eta_0.100/checkpoints/ckpt_010000
python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_mf_vmp_map_spline.py \
                                               --workdir=$WORK_DIR/epidemiology/mean_field/vmp_map_spline \
                                               --config.state_flow_init_path=$WORK_DIR/epidemiology/mean_field/eta_0.100/checkpoints/ckpt_010000
python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_mf_vmp_map_mlp.py \
                                               --workdir=$WORK_DIR/epidemiology/mean_field/vmp_map_mlp \
                                               --config.state_flow_init_path=$WORK_DIR/epidemiology/mean_field/eta_0.100/checkpoints/ckpt_010000
### Neural Spline Flow, initialized on trained eta=0.100
python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_nsf_vmp_map_gp.py \
                                               --workdir=$WORK_DIR/epidemiology/nsf/vmp_map_gp \
                                               --config.state_flow_init_path=$WORK_DIR/epidemiology/nsf/eta_0.100/checkpoints/ckpt_020000
python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_nsf_vmp_map_spline.py \
                                               --workdir=$WORK_DIR/epidemiology/nsf/vmp_map_spline \
                                               --config.state_flow_init_path=$WORK_DIR/epidemiology/nsf/eta_0.100/checkpoints/ckpt_020000
python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_nsf_vmp_map_mlp.py \
                                               --workdir=$WORK_DIR/epidemiology/nsf/vmp_map_mlp \
                                               --config.state_flow_init_path=$WORK_DIR/epidemiology/nsf/eta_0.100/checkpoints/ckpt_020000

# Random Effects
## MCMC
python3 $SMI_DIR/examples/random_effects/main.py --config=$SMI_DIR/examples/random_effects/configs/mcmc_full.py \
                                                 --workdir=$WORK_DIR/random_effects/mcmc/full
python3 $SMI_DIR/examples/random_effects/main.py --config=$SMI_DIR/examples/random_effects/configs/mcmc_cut1.py \
                                                 --workdir=$WORK_DIR/random_effects/mcmc/cut1
python3 $SMI_DIR/examples/random_effects/main.py --config=$SMI_DIR/examples/random_effects/configs/mcmc_cut2.py \
                                                 --workdir=$WORK_DIR/random_effects/mcmc/cut2
### Neural Spline Flow
python3 $SMI_DIR/examples/random_effects/main.py --config=$SMI_DIR/examples/random_effects/configs/flow_nsf_full.py \
                                                 --workdir=$WORK_DIR/random_effects/nsf/full
python3 $SMI_DIR/examples/random_effects/main.py --config=$SMI_DIR/examples/random_effects/configs/flow_nsf_cut1.py \
                                                 --workdir=$WORK_DIR/random_effects/nsf/cut1
python3 $SMI_DIR/examples/random_effects/main.py --config=$SMI_DIR/examples/random_effects/configs/flow_nsf_cut2.py \
                                                 --workdir=$WORK_DIR/random_effects/nsf/cut2
## Variational Meta-Posterior via VMP-map
### Neural Spline Flow
python3 $SMI_DIR/examples/random_effects/main.py --config=$SMI_DIR/examples/random_effects/configs/flow_nsf_vmp_map_mlp.py \
                                                 --workdir=$WORK_DIR/random_effects/nsf/vmp_map_mlp
## Variational Meta-Posterior via VMP-flow
### Neural Spline Flow
python3 $SMI_DIR/examples/random_effects/main.py --config=$SMI_DIR/examples/random_effects/configs/flow_nsf_vmp_flow_mlp.py \
                                                 --workdir=$WORK_DIR/random_effects/nsf/vmp_flow_mlp

