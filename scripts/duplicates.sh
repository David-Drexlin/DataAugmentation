#!/bin/bash
#SBATCH --job-name=PatchCamDup
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2

apptainer run --nv python_container.sif python duplicates.py 
