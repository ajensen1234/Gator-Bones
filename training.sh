#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zachary.gerbi@ufl.edu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6gb
#SBATCH --partition=hpg-ai
#SBATCH --gpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.out

date;hostname;pwd

module load conda
module load itk
module load nvidia


conda activate hpg

python scripts/fit.py config
