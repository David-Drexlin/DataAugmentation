#!/bin/bash
#SBATCH --partition=cpu-2d 
#SBATCH --ntasks-per-node=1

apptainer run --nv -B /home/space/datasets/camelyon16:/home/space/datasets/camelyon16 python_container.sif python save_h5.py
