#!/bin/bash

#BATCH --job-name=wmh_diffuse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=training_fastmri.out
#SBATCH --error=training_fastmri.err
#SBATCH -C a100
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH -A hlp@a100


module purge

module load cuda/12.2.0
module load cudnn/8.9.7.29-cuda

conda init
source ~/.bash_profile
conda activate aoas

cd /lustre/fswork/projects/rech/hlp/uha64uw/diffuse/
python -m examples.mri.training --config examples/mri/configs/config_fastMRI.yaml --continue_training True --parallel True