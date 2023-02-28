#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zachary.gerbi@ufl.edu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6gb
#SBATCH --partition=hpg-ai
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x.%j.out

date;hostname;pwd

module load conda


conda activate hpg

python scripts/fit.py config
