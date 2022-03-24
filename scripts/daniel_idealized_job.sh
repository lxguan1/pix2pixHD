#!/bin/bash

#SBATCH --job-name=idealizeddaniel_job
#SBATCH --mail-user=danonino@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=15g
#SBATCH --ntasks-per-node=1 
#SBATCH --time=00-04:00:00 #commented out, using maximum amount of time available
#SBATCH --account=arbic1
#SBATCH --partition=gpu #Change to debug if want to do debug

# The application(s) to execute along with its input arguments and options: #Load the modules needed, submitted from directory: on the directory, or cd directory (Or have scripts in home and have filepaths to the directories.)


python3 train.py --name idealized_normalize_3 --label_nc 0 --no_instance --dataroot ./datasets/Idealized --no_vgg_loss  --resize_or_crop crop # --no_flip

# To run a job, module load slurm/summit
# sbatch make_pixels1.sh
