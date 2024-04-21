#!/bin/bash
#SBATCH --job-name=RandStain_PatchCam
#SBATCH --partition=gpu-test
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2

apptainer run --nv python_container.sif python train.py --yaml RandStain.yaml --mode test
