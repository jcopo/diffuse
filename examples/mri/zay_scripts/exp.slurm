#!/bin/bash

#SBATCH --job-name=exp_knees
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_exp_knees.out
#SBATCH --error=slurm_logs/%j_exp_knees.err
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH -A hlp@h100

# Set data model (0=WMH, 1=BRATS, 2=fastMRI, 3=brainFastMRI)
data_model_index=4
data_model_array=("WMH" "BRATS" "fastMRI" "brainFastMRI" "kneeFastMRI")
data_model=${data_model_array[$data_model_index]}

# Set model folder name
name_model_folder="models_with_val_deepseek"

# Construct config path
config_path="/lustre/fsn1/projects/rech/hlp/uha64uw/diffuse/data/${data_model}/${name_model_folder}/config_${data_model}.yaml"

# Optional: Print variables for verification
echo "Data Model: $data_model"
echo "Model Folder: $name_model_folder"
echo "Config Path: $config_path"

module purge

module load cuda/12.4.1
module load miniforge/24.9.0
conda activate /lustre/fswork/projects/rech/hlp/uha64uw/.conda/envs/aoas

cd $WORK/diffuse/

# python -m examples.mri.design --rng_key=$2 --num_meas=15  --prefix=$1 --plot --space=$SCRATCH/runs --config "$config_path"
python -m examples.mri.design --rng_key=$SLURM_ARRAY_TASK_ID --num_meas=30  --prefix=$1 --plot --space=$SCRATCH/runs --config "$config_path"