#!/bin/bash

#SBATCH --job-name=idealized_E2_job
#SBATCH --mail-user=lxguan@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=15g
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1 
#SBATCH --time=00-30:00:00 #commented out, using maximum amount of time available
#SBATCH --account=arbic0
#SBATCH --partition=spgpu #Change to debug if want to do debug

# The application(s) to execute along with its input arguments and options: #Load the modules needed, submitted from directory: on the directory, or cd directory (Or have scripts in home and have filepaths to the directories.)

#rm -r ./datasets/Idealized_random/train_A
#rm -r ./datasets/Idealized_random/train_B
#rm -r ./datasets/Idealized_random/test_A
#rm -r ./datasets/Idealized_random/test_B
#mkdir ./datasets/Idealized_random/train_A
#mkdir ./datasets/Idealized_random/train_B
#mkdir ./datasets/Idealized_random/test_A
#mkdir ./datasets/Idealized_random/test_B

#python3 randomize_test_set.py
# --continue_train starts from the latest epoch
for i in {1..700}
do
#    python3 train.py --name idealized_E2 --label_nc 0 --continue_train --no_instance  --dataroot ./datasets/E2 --no_vgg_loss  --resize_or_crop crop  --start_epoch  $i --gan_loss_numpy all_gan_loss_2.npy --disc_loss_numpy all_disc_loss_2.npy
    python3 test.py --name idealized_E2 --resize_or_crop crop --no_instance --which_epoch $i  --label_nc 0 --how_many 300 --dataroot ./datasets/E2 --numpy_file_rmse E2_test_rmse.npy --numpy_file E2_test.npy
    python3 test.py --name idealized_E2 --resize_or_crop crop --no_instance --which_epoch $i --label_nc 0 --how_many 240 --dataroot ./datasets/E2 --phase val --numpy_file_rmse E2_val_rmse.npy --numpy_file E2_val.npy
done
#python3 test.py --name idealized_E2 --resize_or_crop crop --no_instance --which_epoch 700 --label_nc 0 --how_many 100 --dataroot ./datasets/E2 > out.txt

# To run a job, module load slurm/summit
# sbatch make_pixels1.sh

