#!/bin/bash
#SBATCH --account=kgraim
#SBATCH --qos=kgraim
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiayu.huang@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6gb
#SBATCH --partition=hpg-ai
#SBATCH --gpus=a100:4
#SBATCH --time=40:00:00
#SBATCH --output=%x.%j.out

date;hostname;pwd

module load conda

conda activate JTML

# Run a tutorial python script within the container. Modify the path to your container and your script.
python scripts/test.py new_config
