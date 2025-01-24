#!/bin/bash

#BATCH --job-name=train_brats
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_train.out
#SBATCH --error=slurm_logs/%j_train.err
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH -A hlp@a100


module purge

module load cuda/12.2.0
module load cudnn/8.9.7.29-cuda

conda init
source ~/.bash_profile
conda activate aoas

cd $WORK/diffuse/
python -m examples.mri.training --config examples/mri/configs/config_BRATS.yaml --parallel True