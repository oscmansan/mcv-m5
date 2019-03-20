#!/usr/bin/env bash
#SBATCH --job-name fcn8_kitti_transfer
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --qos masterlow
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:Pascal:1
#SBATCH --chdir /home/grupo06/m5-project
#SBATCH --output ../logs/%x_%j.out

source /home/grupo06/venv/bin/activate
python src/main.py --exp_name fcn8_kitti_transfer_${SLURM_JOB_ID} --config_file config/segmentation/fcn8_kitti_transfer.yml