#!/bin/bash
#SBATCH --job-name=pix2pixHD_test
#SBATCH --mail-user=lxguan@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -- mem-per-gpu=16g
#SBATCH --mem-per-cpu=10g
#SBATCH --ntasks-per-node=1
#SBATCH --time=00-02:00:00
#SBATCH --account=arbic1
#SBATCH --partition=gpu
################################ Testing ################################
# labels only
python3 test.py --name label2city_1024p --netG local --ngf 32 --resize_or_crop none $@
