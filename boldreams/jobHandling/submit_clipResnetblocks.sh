#!/bin/bash -l
#SBATCH --account=uludag_gpu
#SBATCH --nodes=1
#SBATCH -p gpu #partition
#SBATCH --gres=gpu:1          # requesting 1 v100 GPU
#SBATCH --cpus-per-task=8         # max 41 CPUs
#SBATCH --mem=32G                 
#SBATCH --time=6:00:00 #wall time  # max 3 days
#SBATCH --mail-user=uzair.hussain@rmp.uhn.ca
#SBATCH --mail-type=ALL #mail notifications

conda activate myenv
python3 /cluster/home/hussaiu/nsd/training_script.py 'RN50x4_clip' 4 5 -1 False

##S#BATCH -C gpu64g                  #specify 32G GPU, rather than 16G
