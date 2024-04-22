import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.RandStainNA.randstainna import RandStainNA

def load_augmentations(augmentation, h5=False): 
    ## h5 indicates whether data is being loaded from ImageFolder or h5
    ## if loaded from h5 the numpy has to be converted back to PIL

    augmentation_mapping = {
        "normalization": normalization(h5),
        "RandStainNA": rand_stain(h5),
        "GeometricAugmentation": geometric_augmentation(h5),
        "ColorAugmentation": color_augmentation(h5),
        }

    if augmentation is None:
        return augmentation_mapping["normalization"]
    
    if augmentation not in augmentation_mapping:
        raise ValueError(f"Augmentation '{augmentation}' not found.")
    
    return augmentation_mapping.get(augmentation, augmentation_mapping["normalization"])

def numpy_to_pil(np_array):
    return Image.fromarray(np_array.astype('uint8'), 'RGB')

def normalization(h5): 
    augmentation_list = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7357, 0.5802, 0.7011], std=[0.2258, 0.2856, 0.2295])
        ## All Normalizations computed on 100K-NoNorm
    ]

    if h5:
        augmentation_list.insert(0, transforms.Lambda(numpy_to_pil))
    
    return transforms.Compose(augmentation_list)

def color_augmentation(h5):
    augmentation_list = [
        transforms.Resize((224, 224)),
        transforms.RandomGrayscale(p=0.2),  
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7357, 0.5802, 0.7011], std=[0.2258, 0.2856, 0.2295])
    ]

    if h5:
        # Insert the Lambda transformation at the beginning if h5 is True
        augmentation_list.insert(0, transforms.Lambda(numpy_to_pil))

    return transforms.Compose(augmentation_list)

def geometric_augmentation(h5):
    augmentation_list = [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7357, 0.5802, 0.7011], std=[0.2258, 0.2856, 0.2295])
    ]

    if h5:
        # Insert the Lambda transformation at the beginning if h5 is True
        augmentation_list.insert(0, transforms.Lambda(numpy_to_pil))

    return transforms.Compose(augmentation_list)

def rand_stain(h5):
    augmentation_list = [
        RandomStainTransform(),
        transforms.ToTensor(),
    ]
    
    if h5:
        augmentation_list.insert(0, transforms.Lambda(numpy_to_pil))
    
    return transforms.Compose(augmentation_list)

class RandomStainTransform:
    def __init__(self):
        self.transforms = [
            RandStainNA(yaml_file="./utils/CRC_HSV.yaml", std_hyper=-0.3, probability=1, distribution="normal", is_train=True),
            RandStainNA(yaml_file="./utils/CRC_HED.yaml", std_hyper=-0.3, probability=1, distribution="normal", is_train=True),
            RandStainNA(yaml_file="./utils/CRC_LAB.yaml", std_hyper=-0.3, probability=1, distribution="normal", is_train=True)
        ]

    def __call__(self, img):
        transform = random.choice(self.transforms)
        ## one of the above color transformation spaces is selected from the list
        return transform(img)
