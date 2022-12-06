#!/bin/bash
#SBATCH --ntasks=1
#SBATCH -J sketch_generation_network
#SBATCH -G 1
#SBATCH -o output/slurm-%j.out
#SBATCH -w devbox4
source ../.bashrc
source ../miniconda3/etc/profile.d/conda.sh
source activate base && "$@"