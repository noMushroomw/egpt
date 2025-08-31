#!/bin/bash
#SBATCH --nodes=1
#SBATCH --constraint=el9
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --partition=hpg-b200
#SBATCH --time=128:00:00
#SBATCH --job-name=mae
#SBATCH --mem=70GB
### GPU options ###
#SBATCH --gpus-per-node=8


module load conda
conda activate egpt

srun python xxx