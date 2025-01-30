#!/bin/bash

#SBATCH --job-name=exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_exp.out
#SBATCH --error=slurm_logs/%j_exp.err
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH -A hlp@h100

# Set data model (0=WMH, 1=BRATS, 2=fastMRI)
data_model_index=1
data_model_array=("WMH" "BRATS" "fastMRI")
data_model=${data_model_array[$data_model_index]}

# Set model folder name
name_model_folder="models_with_val"

# Construct config path
config_path="/lustre/fsn1/projects/rech/hlp/uha64uw/diffuse/data/${data_model}/${name_model_folder}/config_${data_model}.yaml"

# Optional: Print variables for verification
echo "Data Model: $data_model"
echo "Model Folder: $name_model_folder"
echo "Config Path: $config_path"

set -x

#srun python -m examples.mri.design --rng_key=$2 --num_meas=15  --prefix=$1 --plot --space=$SCRATCH/runs --config "$config_path"
srun python -m examples.mri.design --rng_key=$SLURM_ARRAY_TASK_ID --num_meas=15  --prefix=$1 --plot --space=$SCRATCH/runs --config "$config_path"