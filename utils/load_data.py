import h5py
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np

class HDF5Dataset_Labels(Dataset):
    def __init__(self, file_path, dataset_name=None):
        self.file_path = file_path
        self.dataset_name = dataset_name

        with h5py.File(self.file_path, 'r') as file:
            if self.dataset_name:
                # Handling named datasets
                self.dataset_len = len(file[self.dataset_name])
            else:
                # Handling key-indexed datasets
                self.keys = list(file.keys())
                self.dataset_len = len(self.keys)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as file:
            if self.dataset_name:
                # Access named dataset
                data = file[self.dataset_name][idx]
            else:
                # Access key-indexed dataset
                key = self.keys[idx]
                data = file[key][()]

            # Assuming 'data' is a single label or an array of labels
            # If 'data' is a single label, this operation will work directly.
            # If 'data' is an array and only the second column has labels, then use data = data[1] - 1
            # Here we adjust labels from 1-based to 0-based.
            adjusted_label = data[1] - 1

            return adjusted_label


class HDF5Dataset(Dataset):
    def __init__(self, file_path, transform,  dataset_name=None, train=True):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.transform = transform
        self.train = train

        with h5py.File(self.file_path, 'r') as file:
            if self.dataset_name:
                # Handling named datasets
                self.dataset_len = len(file[self.dataset_name])
            else:
                # Handling key-indexed datasets
                self.keys = list(file.keys())
                self.dataset_len = len(self.keys)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as file:
            if self.dataset_name:
                # Access named dataset
                data = file[self.dataset_name][idx]
            else:
                # Access key-indexed dataset
                key = self.keys[idx]
                data = file[key][()]

        # Apply transformations
        if self.transform:
            data = self.transform(data)

        # Convert to tensor if training
        if self.train:
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            elif isinstance(data, Image.Image):
                data = transforms.ToTensor()(data)

        return data.clone().detach()

        
class MergedDataset(Dataset):
    def __init__(self, data_dataset, labels_dataset, is_test=False):
        # Check if data_dataset is a tuple (indicating multiple datasets)
        self.is_multiple = isinstance(data_dataset, tuple)

        self.data_dataset = data_dataset
        self.labels_dataset = labels_dataset
        self.is_test = is_test

        # Ensure the lengths match
        if self.is_multiple:
            assert all(len(data) == len(labels_dataset) for data in data_dataset), "Datasets must be of equal length"
        else:
            assert len(data_dataset) == len(labels_dataset), "Datasets must be of equal length"

    def __len__(self):
        if self.is_multiple:
            return len(self.data_dataset[0])
        else:
            return len(self.data_dataset)

    def __getitem__(self, idx):
        if self.is_multiple:
            # Process multiple datasets
            data = tuple(dataset[idx] for dataset in self.data_dataset)
        else:
            # Process single dataset
            data = self.data_dataset[idx]

        label = self.labels_dataset[idx]

        if self.is_test:
            return (*data, label, idx) if self.is_multiple else (data, label, idx)
        else:
            return (*data, label) if self.is_multiple else (data, label)
