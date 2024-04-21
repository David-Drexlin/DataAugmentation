import torchvision.transforms as transforms
from utils.RandStainNA.randstainna import RandStainNA
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np

def load_augmentations(augmentation, h5=False): 
    augmentation_mapping = {
        "normalization": normalization(),
        "RandStainNA": RandStain(),
        "GeometricAugmentation": GeometricAugmentation(),
        "ColorAugmentation": ColorAugmentation(h5),
        "FancyPCA": AlexNet(), 
    }
    
    if augmentation is None:
        return augmentation_mapping["normalization"]
    
    if augmentation not in augmentation_mapping:
        raise ValueError(f"Augmentation '{augmentation}' not found.")
    
    return augmentation_mapping[augmentation]


def numpy_to_pil(np_array):
    # Assuming the numpy array is an image with shape (H, W, C)
    return Image.fromarray(np_array.astype('uint8'), 'RGB')


def normalization(): 
    return transforms.Compose([
        transforms.Lambda(numpy_to_pil),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7357, 0.5802, 0.7011], std=[0.2258, 0.2856, 0.2295])
    # PCAM 
    # transforms.Normalize(mean=[0.7008, 0.5384, 0.6916], std=[0.2350, 0.2774, 0.2129])
    # std for 224, 224 ->>> transforms.Normalize(mean=[0.7008, 0.5384, 0.6916], std=[0.2177, 0.2621, 0.1947]) 
])

def AlexNet():
    return transforms.Compose([
        transforms.Lambda(numpy_to_pil),
        transforms.Resize((256, 256)),            
        transforms.RandomResizedCrop((224, 224)), 
        FancyPCA(alpha_std=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7357, 0.5802, 0.7011], std=[0.2258, 0.2856, 0.2295])
    ])

class FancyPCA(object):
    def __init__(self, alpha_std=0.1):
        self.alpha_std = alpha_std
        #  precomputed eigenvectors and eigenvalues for No-Norm-CRC
        self.eig_vecs = np.array([[ 5.73437021e-01,  5.85066269e-01,  5.73469653e-01],
                                  [ 7.07113062e-01,  2.71266448e-05, -7.07100500e-01],
                                  [-4.13716208e-01,  8.10985487e-01, -4.13692445e-01]])
        self.eig_vals = np.array([0.17916879, 0.00572009, 0.00093162])

    def __call__(self, img):
        if img.mode != 'RGB':
            raise ValueError('FancyPCA only applies to RGB images')

        img = np.array(img, dtype=np.float64)  # Convert to float64

        alpha = np.random.normal(0, self.alpha_std, size=(3,))
        adjustment = np.dot(self.eig_vecs, self.eig_vals * alpha)

        for i in range(3):  # for each channel
            img[..., i] += adjustment[i]

        # Clip to ensure the values stay in the 0-255 range
        img = np.clip(img, 0, 255).astype(np.uint8)

        return Image.fromarray(img)

class RandomStainTransform:
    def __init__(self):
        self.transforms = [
            RandStainNA(yaml_file="./utils/CRC_HSV.yaml", std_hyper=-0.3, probability=1, distribution="normal", is_train=True),
            RandStainNA(yaml_file="./utils/CRC_HED.yaml", std_hyper=-0.3, probability=1, distribution="normal", is_train=True),
            RandStainNA(yaml_file="./utils/CRC_LAB.yaml", std_hyper=-0.3, probability=1, distribution="normal", is_train=True)
        ]

    def __call__(self, img):
        transform = random.choice(self.transforms)
        return transform(img)

def RandStain():
    return transforms.Compose([
        transforms.Lambda(numpy_to_pil),
        RandomStainTransform(),
        transforms.ToTensor(),
        # Additional transforms (if any) can go here.
    ])


class ClassGeometricAugmentation:
    def __init__(self):
        self.transformations = [
            transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(size=(224,224), scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC)
            ]),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15)], p=1)
        ]

    def __call__(self, image):
        transform = random.choice(self.transformations)
        return transform(image)

    def __repr__(self):
        return self.__class__.__name__ + '()'

# Usage in a Compose

def GeometricAugmentation():
    return transforms.Compose([
    transforms.Lambda(numpy_to_pil),  # Assuming this is a custom function you've defined
    transforms.Resize((224, 224)),
    ClassGeometricAugmentation(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7357, 0.5802, 0.7011], std=[0.2258, 0.2856, 0.2295])
])

def ColorAugmentation(h5=False):
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

