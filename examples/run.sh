#!/bin/bash
set -e
set -x

# Assume we are located at the SMI directory
SMI_DIR=$PWD

# Directory to save all outputs
WORK_DIR=$HOME/modularbayes-output

# Create output directory and install missing dependencies
mkdir -p $WORK_DIR

## Epidemiological Model ###

## One posterior for each eta
all_eta=('0.001' '0.100' '0.200' '0.500' '1.000')

## MCMC
for eta in "${all_eta[@]}"
do
  python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/mcmc.py \
                                                 --workdir=$WORK_DIR/epidemiology/mcmc/eta_$eta \
                                                 --config.smi_eta_cancer=$eta \
                                                 --log_dir $WORK_DIR/epidemiology/mcmc/eta_$eta/log_dir \
                                                 --alsologtostderr
done

## Mean Field Variational Inference (MFVI)
for eta in "${all_eta[@]}"
do
  python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_mf.py \
                                                 --workdir=$WORK_DIR/epidemiology/mean_field/eta_$eta \
                                                 --config.smi_eta_cancer=$eta \
                                                 --log_dir $WORK_DIR/epidemiology/mean_field/eta_$eta/log_dir \
                                                 --alsologtostderr
done

### Neural Spline Flow
for eta in "${all_eta[@]}"
do
  python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_nsf.py \
                                                 --workdir=$WORK_DIR/epidemiology/nsf/eta_$eta \
                                                 --config.smi_eta_cancer=$eta \
                                                 --log_dir $WORK_DIR/epidemiology/nsf/eta_$eta/log_dir \
                                                 --alsologtostderr
done

# ## Variational Meta-Posterior via VMP-map (Mean field)
# python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_mf_vmp_map.py \
#                                                --workdir=$WORK_DIR/epidemiology/mean_field/vmp_map \
#                                                --log_dir $WORK_DIR/epidemiology/mean_field/vmp_map/log_dir \
#                                                --alsologtostderr

# ## Variational Meta-Posterior via VMP-map (Neural Spline Flow)
# python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_nsf_vmp_map.py \
#                                                --workdir=$WORK_DIR/epidemiology/nsf/vmp_map \
#                                                --log_dir $WORK_DIR/epidemiology/nsf/vmp_map/log_dir \
#                                                --alsologtostderr

## Variational Meta-Posterior via VMP-flow (Neural Spline Flow)
### Neural Spline Flow
python3 $SMI_DIR/examples/epidemiology/main.py --config=$SMI_DIR/examples/epidemiology/configs/flow_nsf_vmp_flow.py \
                                               --workdir=$WORK_DIR/epidemiology/nsf/vmp_flow \
                                               --log_dir $WORK_DIR/epidemiology/nsf/vmp_flow/log_dir \
                                               --alsologtostderr



### Random Effects Model ###

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
python3 $SMI_DIR/examples/random_effects/main.py --config=$SMI_DIR/examples/random_effects/configs/flow_nsf_cut3.py \
                                                 --workdir=$WORK_DIR/random_effects/nsf/cut3
# ## Variational Meta-Posterior via VMP-map (Neural Spline Flow)
# python3 $SMI_DIR/examples/random_effects/main.py --config=$SMI_DIR/examples/random_effects/configs/flow_nsf_vmp_map.py \
#                                                  --workdir=$WORK_DIR/random_effects/nsf/vmp_map
## Variational Meta-Posterior via VMP-flow (Neural Spline Flow)
python3 $SMI_DIR/examples/random_effects/main.py --config=$SMI_DIR/examples/random_effects/configs/flow_nsf_vmp_flow.py \
                                                 --workdir=$WORK_DIR/random_effects/nsf/vmp_flow

