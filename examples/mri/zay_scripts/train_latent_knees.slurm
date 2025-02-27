#!/bin/bash

#SBATCH --job-name=dsknee
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_train.out
#SBATCH --error=slurm_logs/%j_train.err
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH -A hlp@h100


module purge

module load cuda/12.4.1
module load miniforge/24.9.0
conda activate /lustre/fswork/projects/rech/hlp/uha64uw/.conda/envs/aoas

cd $WORK/diffuse/
python -m examples.mri.python_script.training --config examples/mri/configs/config_latent_kneeFastMRI.yaml --latent True