#!/usr/bin/env python
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from Master.utils.utils import no_augmentations, HDF5Dataset
import yaml
import argparse
import numpy as np
import random

yaml_file_path = './hyper.yaml'
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def compute_data_normalizations(path):
    with open(yaml_file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)

    batch_size = hyperparameters['batch_size']

    data = HDF5Dataset(path, 'x', no_augmentations())
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Variables to store sum and square sum of pixel values, initialized on the device
    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)

    # Calculate the sum and square sum
    for images in tqdm(loader):
        images = images.to(device)
        mean += images.mean([0, 2, 3])
        std += images.std([0, 2, 3])

    # Average over the number of batches
    mean /= len(loader)
    std /= len(loader)

    return (mean.cpu() , std.cpu())

def main():
    parser = argparse.ArgumentParser(description='Compute dataset normalization values.')
    parser.add_argument('data_path', type=str, help='Path to the dataset file')
    args = parser.parse_args()

    mean, std = compute_data_normalizations(args.data_path)
    print(f"Mean: {mean}\nStandard Deviation: {std}")

if __name__ == "__main__":
    main()