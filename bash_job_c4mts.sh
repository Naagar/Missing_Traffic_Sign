#!/bin/bash
#SBATCH -A research
#SBATCH --job-name=c4mts
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3000
#SBATCH --output=output_files/task_1_detection__yono8l%j.txt       # Output file.
#SBATCH --mail-type=END                # Enable email about job finish
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail
#SBATCH --time=4-00:00:00

# module load cudnn/7-cuda-10.2
# module load cudnn/7.6-cuda-10.0
# source activate ff_venv 		# to activate the conda virtual environment 
# source venv/bin/activate # to activate the python virtual environment

# rm -r /scratch/sandeep.nagar
# mkdir /scratch/sandeep.nagar
# rsync -aP sandeep.nagar@ada.iiit.ac.in:/share1/sandeep.nagar/celeba/train.tar /scratch/sandeep.nagar
# rsync -aP sandeep.nagar@ada.iiit.ac.in:/share1/sandeep.nagar/celeba/validation.tar /scratch/sandeep.nagar

# scp ada:/share1/$aditya.kalappa/train /scratch
# comment # cinc_cuda_level1
# fastflow/layers/conv

# cd fastflow2
source activate c4mts
echo "Training the model"
python c4mts_task_1.py
# print(finisg)
# python fastflow_cifar.py

