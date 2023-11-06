#!/bin/bash
#SBATCH --job-name=all_data_training
#SBATCH --mail-user=ajensen123@ufl.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output ./slurm/logs/my_job-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5gb
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# module load cuda/11.1.0
module load gcc
export PATH=/blue/banks/ajensen123/JTML/envs/jtml_swin_env/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/blue/banks/ajensen123/JTML/envs/jtml_swin_env/lib/

python ./scripts/fit.py hum_config
