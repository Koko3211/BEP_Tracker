#!/bin/bash
#SBATCH -p gpu_mig
#SBATCH -N 1
#SBATCH -J yolo_ByteTrack
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=START,END
#SBATCH --mail-user=MailAddress
#SBATCH --output=slurm_output_ByteTrack%A.txt
#SBATCH --error=slurm_error_ByteTrack%A.txt
#SBATCH --reservation=terv92681

module load 2025
source "$HOME/.venv/bin/activate"

cd $HOME/train

srun python $HOME/train/Train_Yolo.py 
