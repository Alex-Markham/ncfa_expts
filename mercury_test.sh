#!/bin/bash

#SBATCH --account=pi-naragam
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --time=5-05:00:00

#SBATCH --job-name=snakemake

module load python/booth/3.12
cd ~/ncfa_expts/
source .venv/bin/activate

snakemake --profile ./ all
