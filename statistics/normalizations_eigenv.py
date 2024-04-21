#!/usr/bin/env python
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from utils.augmentations import no_augmentations
import yaml
import argparse
from sklearn.decomposition import IncrementalPCA
import numpy as np
import random

yaml_file_path = './utils/yaml/hyper.yaml'
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform

        self.keys = []  # List to store group and dataset names
        with h5py.File(self.file_path, 'r') as file:
            for group_name in file.keys():
                group = file[group_name]
                for dataset_name in group.keys():
                    self.keys.append((group_name, dataset_name))  # Store tuple of group and dataset name

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        group_name, dataset_name = self.keys[index]  # Unpack group and dataset names

        with h5py.File(self.file_path, 'r') as file:
            data = file[group_name][dataset_name][()]  # Access the dataset within the group

        if self.transform is not None:
            data = self.transform(data)

        return data.clone().detach()


def compute_data_normalizations(path):
    with open(yaml_file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)

    batch_size = hyperparameters['batch_size']

    data = HDF5Dataset(path, no_augmentations())
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

def compute_pca(loader, num_samples=10000):
    ipca = IncrementalPCA(n_components=3)

    for i, images in enumerate(tqdm(loader)):
        # Reshape images: From [batch_size, channels, height, width] to [batch_size, height*width, channels]
        reshaped_images = images.permute(0, 2, 3, 1).view(images.size(0), -1, 3).numpy()

        # Flatten the batch for PCA
        flat_images = reshaped_images.reshape(-1, 3)
        
        # Partial fit on the flattened images
        ipca.partial_fit(flat_images)

    eig_vecs = ipca.components_.T
    eig_vals = ipca.explained_variance_

    return eig_vecs, eig_vals


def main():
    parser = argparse.ArgumentParser(description='Compute dataset normalization values.')
    parser.add_argument('data_path', type=str, help='Path to the dataset file')
    args = parser.parse_args()

    with open(yaml_file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)

    batch_size = hyperparameters['batch_size']
    data = HDF5Dataset(args.data_path, no_augmentations())
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    mean, std = compute_data_normalizations(args.data_path)
    print(f"Mean: {mean}\nStandard Deviation: {std}")

    # Compute PCA
    eig_vecs, eig_vals = compute_pca(loader)
    print(eig_vecs)
    print(eig_vals)
    f = open("eigen.txt", "a")

    eig_vecs_str = np.array2string(eig_vecs)
    eig_vals_str = np.array2string(eig_vals)
    mean = np.array2string(np.array(mean))
    std = np.array2string(np.array(std))
    # Write the string to the file
    f.write("Eigenvectors: " + eig_vecs_str + "\n")
    f.write("Eigenvalues: " + eig_vals_str + "\n")
    f.write("Mean: " + mean + "\n")
    f.write("Std: " + std + "\n")
    f.close()

if __name__ == "__main__":
    main()
