#!/bin/bash
#SBATCH --job-name=LARS_AN
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2

apptainer run --nv python_container.sif python train.py --yaml LARS_AN.yaml
