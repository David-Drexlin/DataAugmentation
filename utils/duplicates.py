import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import torchvision.transforms as transforms
from PIL import Image

class HDF5Dataset(Dataset):
    def __init__(self, file_path, dataset_name, transform=None):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.transform = transform
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file[self.dataset_name])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
            with h5py.File(self.file_path, 'r') as file:
                data = file[self.dataset_name][idx]

            # Apply transformations
            if self.transform:
                data = self.transform(data)

            return torch.tensor(data), idx

import matplotlib.pyplot as plt

def plot_image(tensor_img, title=""):
    # Convert the tensor to numpy array and transpose the axes from (C, H, W) to (H, W, C)
    np_img = tensor_img.numpy().transpose(1, 2, 0)
    plt.imshow(np_img)
    plt.title(title)
    plt.axis('off')
    plt.show()


def numpy_to_pil(np_array):
    # Assuming the numpy array is an image with shape (H, W, C)
    return Image.fromarray(np_array.astype('uint8'), 'RGB')

transform = transforms.Compose([
    transforms.Lambda(lambda x: numpy_to_pil(x)),
    transforms.ToTensor(),
])

def check_duplicates_in_batches(loader):
    batch_duplicate_info = []

    for batch_idx, (data, indices) in enumerate(loader):
        num_images = len(data)
        duplicate_pairs = set()  # To store unique pairs of duplicates

        # Compare each image with every other image in the batch
        for i in range(num_images):
            for j in range(i + 1, num_images):
                if torch.equal(data[i], data[j]):
                    duplicate_pairs.add((indices[i].item(), indices[j].item()))  # Use global indices

        # Compute the number of unique images that have duplicates
        unique_images_with_duplicates = set(idx for pair in duplicate_pairs for idx in pair)

        # Calculate the percentage of images that are duplicates
        if num_images > 0:
            percentage_duplicates = (len(unique_images_with_duplicates) / num_images) * 100
        else:
            percentage_duplicates = 0

        batch_duplicate_info.append({
            "batch_idx": batch_idx,
            "duplicate_pairs": duplicate_pairs,
            "percentage_duplicates": percentage_duplicates
        })
    return batch_duplicate_info

def count_transitive_depth(pairs, start, depth=0, visited=None):
    """
    Count the transitive depth starting from a given number.

    :param pairs: List of tuples representing the pairs.
    :param start: The starting number for which to count the depth.
    :param depth: Current depth (used in recursive calls).
    :param visited: Set of visited nodes to prevent infinite loops.
    :return: Maximum depth found from the start.
    """
    if visited is None:
        visited = set()

    # Avoid re-visiting the same node
    if start in visited:
        return depth
    visited.add(start)

    max_depth = depth
    for x, y in pairs:
        if x == start:
            current_depth = count_transitive_depth(pairs, y, depth + 1, visited.copy())
            max_depth = max(max_depth, current_depth)

    return max_depth

def unique_starting_points(pairs):
    """
    Identify unique starting points from a list of pairs.

    :param pairs: List of tuples representing the pairs.
    :return: Set of unique starting points.
    """
    all_firsts = {a for a, _ in pairs}
    all_seconds = {b for _, b in pairs}
    return all_firsts - all_seconds

def get_images_by_indices(loader, indices):
    images = []
    cumulative_index = 0  # Keeps track of the index across all batches

    for batch in loader:
        data, _ = batch
        batch_size = len(data)

        for index in indices:
            if cumulative_index <= index < cumulative_index + batch_size:
                # Adjust the index for the current batch
                adjusted_index = index - cumulative_index
                images.append(data[adjusted_index])

        cumulative_index += batch_size

    return images


def main():
    hdf5_dataset = HDF5Dataset('./data/pcam/camelyonpatch_level_2_split_test_x.h5', 'x', transform=transform)
    loader = DataLoader(hdf5_dataset, batch_size=32770, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    batch_duplicate_info = check_duplicates_in_batches(loader)
    
    f = open("duplicates.txt", "a")
    for info in batch_duplicate_info:
        
        f.write(f"Batch {info['batch_idx']}: {info['percentage_duplicates']:.2f}% duplicates" + "\n")
        f.write(f"Number of Duplicates: {len(info['duplicate_pairs'])}" + "\n")

        print(f"Batch {info['batch_idx']}: {info['percentage_duplicates']:.2f}% duplicates")
        print(f"Number of Duplicates: {len(info['duplicate_pairs'])}")


    starts = unique_starting_points(info['duplicate_pairs'])
    
    depths = []
    for start in starts:
        depth = count_transitive_depth(info['duplicate_pairs'], start)
        depths.append(depth)

        f.write(f"Depth for {start}: {depth}" + "\n")
        print(f"Depth for {start}: {depth}")

    unique_count = sum(depth == 1 for depth in depths)
    total_elements = len(depths)
    percentage_unique = (unique_count / total_elements) * 100

    print(f"Percentage of images appearing more than single duplicates: {100-percentage_unique}%")
    f.write(f"Percentage of images appearing more than single duplicates: {100-percentage_unique}%")

    plt.hist(depths, bins=range(min(depths), max(depths) + 2), align='left', rwidth=0.8)


    plt.xlabel('How many instances of the same image appear in the dataset')
    plt.savefig('duplicates_hist.png')
    f.close()
    
    images = get_images_by_indices(loader, starts)
    fig, axs = plt.subplots(1, len(starts), figsize=(5 * len(starts), 5))

    for i, idx in enumerate(starts):
        image = images[i]
        axs[i].imshow(image.permute(1, 2, 0)) # Assuming images are in CxHxW format
        axs[i].set_title(f"\nDuplicates: {depths[i]} at {idx}")
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig('duplicates.png')


if __name__ == "__main__":
    main()
