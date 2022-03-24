#!/bin/bash

#SBATCH --job-name=test_wreal_job
#SBATCH --mail-user=danonino@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=15g
#SBATCH --ntasks-per-node=1
#SBATCH --time=00-02:00:00 #commented out, using maximum amount of time available
#SBATCH --account=arbic1
#SBATCH --partition=gpu #Change to debug if want to do debug

# The application(s) to execute along with its input arguments and options: #Load the modules needed, submitted from directory: on the directory, or cd directory (Or have scripts in h$


python3 test.py --name idealized_normalize_4  --label_nc 0 --no_instance --dataroot ./datasets/Idealized_test  --results_dir ./results2/  --resize_or_crop crop  --no_flip --how_many 10 --which_epoch 200  --phase test
#python3 test.py --name idealized --label_nc 0 --no_instance --dataroot ./datasets/Idealized   --resize_or_crop crop  --no_flip --how_many 10 --which_epoch 150  --phase val
#python3 test.py --name idealized --label_nc 0 --no_instance --dataroot ./datasets/Idealized   --resize_or_crop crop  --no_flip --how_many 10 --which_epoch 100  --phase val
#python3 test.py --name idealized --label_nc 0 --no_instance --dataroot ./datasets/Idealized   --resize_or_crop crop  --no_flip --how_many 10 --which_epoch 50  --phase val
#python3 test.py --name idealized --label_nc 0 --no_instance --dataroot ./datasets/Idealized   --resize_or_crop crop  --no_flip --how_many 10 --which_epoch 10  --phase val



# To run a job, module load slurm/summit
# sbatch make_pixels1.sh
