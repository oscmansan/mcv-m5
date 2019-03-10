#!/usr/bin/env bash
#SBATCH --job-name oscarnet_kitti
#SBATCH --ntasks 4
#SBATCH --mem 16G
#SBATCH --priority mhigh
#SBATCH --partition mhigh,mlow
#SBATCH --gres gpu:1
#SBATCH --chdir /home/grupo06/.oscar/m5-project
#SBATCH --output ../logs/%x_%u_%j.out

source /home/grupo06/venv/bin/activate
python src/main.py --exp_name oscarnet_kitti_${SLURM_JOB_ID} --config_file config/oscarnet_kitti.yml