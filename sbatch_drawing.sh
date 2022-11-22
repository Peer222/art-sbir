#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -J resnet_sketchy_train
#SBATCH -G 1
#SBATCH -o output/slurm-%j.out
#SBATCH -w devbox5
source ../.bashrc
source ../miniconda3/etc/profile.d/conda.sh
source env create -f drawing_utils/environment.yml
source activate drawings && "$@"