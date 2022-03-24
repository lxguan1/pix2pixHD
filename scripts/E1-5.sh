#!/bin/bash
#SBATCH --job-name=idealized_E1-5_job
#SBATCH --mail-user=lxguan@umich.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=2g
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00-00:10:00 #commented out, using maximum amount of time available
#SBATCH --account=arbic0
#SBATCH --partition=standard #Change to debug if want to do debug
sbatch ./scripts/idealized_E1_job.sh
sbatch ./scripts/idealized_E2_job.sh
sbatch ./scripts/idealized_E3_job.sh
sbatch ./scripts/idealized_E4_job.sh
sbatch ./scripts/idealized_E5_job.sh
