#!/bin/bash
#SBATCH --partition=gpu-5h
#SBATCH --ntasks-per-node=1

apptainer run python_container.sif python normalizations.py /home/daviddrexlin/Master/data/p16/train_images.h5
