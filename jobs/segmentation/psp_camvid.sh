#!/usr/bin/env bash
#SBATCH --job-name psp_camvid
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --qos masterlow
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/.oscar/m5-project
#SBATCH --output ../logs/%x_%j.out

source /home/grupo06/venv/bin/activate
python src/main.py --exp_name psp_camvid_${SLURM_JOB_ID} --config_file config/segmentation/psp_camvid.yml