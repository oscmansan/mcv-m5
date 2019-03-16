#!/usr/bin/env bash
#SBATCH --job-name fcn8_camvid
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --qos masterhigh
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/m5-project
#SBATCH --output ../logs/%x_%j.out

source /home/grupo06/venv/bin/activate
python src/main.py --exp_name fcn8_camvid_${SLURM_JOB_ID} --config_file config/segmentation/fcn8_camvid.yml