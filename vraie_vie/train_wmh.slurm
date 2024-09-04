#!/bin/bash

#BATCH --job-name=wmh_diffuse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=sliced.out
#SBATCH --error=sliced.err
#SBATCH -C a100
#SBATCH -A hlp@a100


module purge

module load cuda/12.2.0
module load cudnn/8.9.7.29-cuda

conda init
source ~/.bash_profile
conda activate aoas

python -u /lustre/fswork/projects/rech/hlp/uha64uw/aistat24/diffuse/vraie_vie/train_wmh.py