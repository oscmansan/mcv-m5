#!/usr/bin/env bash
#SBATCH --job-name w3
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/m5-project
#SBATCH --output ../logs/%x_%u_%j.out

source /home/grupo06/venv/bin/activate
python src/main.py --exp_name seg_fcn8_kitti${SLURM_JOB_ID} --config_file config/semantic_segmentation/SemSeg_sample_fcn8_kitti.yml