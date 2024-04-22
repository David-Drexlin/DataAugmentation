import os
import yaml
import pickle
import numpy as np
import torch
import timm
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchgeo.models import ResNet18_Weights
from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient

def load_yaml(filename):
    yaml_file_path = './utils/yaml/'+filename 
    with open(yaml_file_path, 'r') as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters

def load_model(model_name, num_classes, pretrained=False, kind="image_net"):
    """
    Loads a specified machine learning model with the option to initialize with pretrained weights.
    This function supports various models including ResNet and Vision Transformers, and allows for 
    custom pretrained weight initialization depending on the model and specified `kind`.

    Parameters:
    - model_name (str): The name of the model to load. Supported models include 'resnet18', 'resnet34', 
                        'resnet50', and 'vit_small_patch16_224'.
    - num_classes (int): The number of output classes for the model. This modifies the final layer 
                         to match the specified number of classes.
    - pretrained (bool): If True, initializes the model with pretrained weights according to the `kind` parameter.
                         If False, the model will initialize weights randomly.
    - kind (str): A string that specifies the type of pretrained weights to use. Options include 'image_net' 
                  for ImageNet pretrained weights, and 'RMS' for remote sensing data pretrained weights. 
                  Default is 'image_net'.

    Returns:
    - model: The loaded model with weights initialized as specified, and adjusted to the correct number of output classes.

    Raises:
    - ValueError: If the specified model_name is not recognized or supported.

    The function is designed to be flexible to accommodate different types of pretrained weights and model architectures,
    making it suitable for a variety of machine learning tasks.
    """

    model_mapping = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "vit_small_patch16_224": "vit_small_patch16_224",
        # Add other models
    }

    model_weight_mapping = {
        "resnet18": None,
        "resnet34": None,
        "resnet50": None,
        "vit_small_patch16_224": False,  # Update as needed
    }

    if pretrained: # if false random weight initalization, otherwise update the None values
        if kind == "image_net":
            model_weight_mapping.update({
                "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
                "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
                "resnet50": models.ResNet50_Weights.IMAGENET1K_V1,
                "vit_small_patch16_224": True,
            })
            # RMS = remote sensing pretrained weights
        elif kind == "RMS":
            model_weight_mapping.update({
                "resnet18": ResNet18_Weights.SENTINEL2_RGB_MOCO,  # Update this appropriately
            })

    model_constructor = model_mapping.get(model_name)
    weights = model_weight_mapping.get(model_name)

    if model_constructor is None:
        raise ValueError(f"Model {model_name} not recognized")

    elif kind == "RMS": 
        model = timm.create_model("resnet18", num_classes=num_classes)
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

    elif "vit" in model_name:
        # Handle loading of Vision Transformer models
        model = timm.create_model(model_name, pretrained=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        # Handle loading of other models
        model = model_constructor(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def load_path(dataset_type):
    # loads the h5 paths to model.py if used h5 is used as data-source

    if dataset_type == 'crc':
        # CRC dataset path
        train_data_file = '/home/daviddrexlin/Master/data/crc/train_crc_images.h5'
        train_label_file = '/home/daviddrexlin/Master/data/crc/train_crc_labels.h5'
        valid_data_file = '/home/daviddrexlin/Master/data/crc/valid_crc_images.h5'
        valid_label_file = '/home/daviddrexlin/Master/data/crc/valid_crc_labels.h5'
        test_data_file = '/home/daviddrexlin/Master/data/crc/test_crc_images.h5'
        test_label_file = '/home/daviddrexlin/Master/data/crc/test_crc_labels.h5'

    elif dataset_type == 'pcam':
        # PCam dataset paths
        train_data_file = './data/pcam/camelyonpatch_level_2_split_train_x.h5-002'
        train_label_file = './data/pcam/camelyonpatch_level_2_split_train_y.h5'
        valid_data_file = './data/pcam/camelyonpatch_level_2_split_valid_x.h5'
        valid_label_file = './data/pcam/camelyonpatch_level_2_split_valid_y.h5'
        test_data_file = './data/pcam/camelyonpatch_level_2_split_test_x.h5'
        test_label_file = './data/pcam/camelyonpatch_level_2_split_test_y.h5'

    elif dataset_type == 'camelyon':
        # Camelyon dataset paths
        train_data_file = '/home/daviddrexlin/Master/data/train_images.h5'
        train_label_file = '/home/daviddrexlin/Master/data/train_labels.h5'
        valid_data_file = '/home/daviddrexlin/Master/data/valid_images.h5'
        valid_label_file = '/home/daviddrexlin/Master/data/valid_labels.h5'
        test_data_file = '/home/daviddrexlin/Master/data/test_images.h5'
        test_label_file = '/home/daviddrexlin/Master/data/test_labels.h5'

    else: 
        train_data_file = None
        train_label_file = None
        valid_data_file = None
        valid_label_file = None
        test_data_file = None
        test_label_file = None

    return {
        "train_data_file": train_data_file,
        "train_label_file": train_label_file,
        "valid_data_file": valid_data_file,
        "valid_label_file": valid_label_file,
        "test_data_file": test_data_file,
        "test_label_file": test_label_file
    }

def weight_decay_filter(param):
    # Apply weight decay to all parameters except bias and batch normalization layers
    if param.ndim == 1 or 'bn' in param.name:
        return False
    return True

#### From https://github.com/facebookresearch/dino/blob/main/utils.py
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

#### From https://github.com/facebookresearch/dino/blob/main/utils.py
class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        loss = None
        if closure is not None:
            loss = closure()

        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])
        
    #return loss

def LRP(model, pkl_path, save_dir):
    ## Given a list of images paths in a Pickle file, LRP is applied to
    ## the images contained in the pickle, currently not (yet) used

    model.eval()

    # Load prediction differences
    with open(pkl_path, 'rb') as f:
        prediction_differences = pickle.load(f)
    
    difference, index, data, true_label = zip(*prediction_differences)
    true_label = torch.tensor(true_label).unsqueeze(1)
    data = torch.stack(data)
    data.requires_grad = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = data.to(device)
    true_label = true_label.to(device)

    # Create the subdirectory
    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonGammaBox(low=-3., high=3., canonizers=canonizers)

    # Use the Gradient attributor with the composite
    with Gradient(model=model, composite=composite) as attributor:
        out, relevance = attributor(data, true_label)

        # Normalize the relevance scores to [0, 1]
        # Sum over the color channel and convert to numpy array
        # This sums the relevance scores across the channel dimension. If your data is an image tensor in the format (batch_size, channels, height, width), this operation collapses the channels (e.g., RGB channels in an image), resulting in a 2D representation of relevance for each image in the batch.
        relevance = relevance.sum(1).squeeze().cpu().detach().numpy()
        #This line normalizes the relevance scores to be within the range [0, 1].
        relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min())

    for i in range(len(prediction_differences)):
        # Create file names for the data image and the heatmap
        data_filename = './LRP/'+save_dir+f"/test_image__explained_{i}_{index[i]}.png"
        heatmap_filename =  './LRP/'+save_dir+f"/heatmap_{i}.png"

        # data_filename = os.path.join(save_dir, f"/test_image_{index[i]}_explained_{i}.png")
        # heatmap_filename = os.path.join(save_dir, f"/heatmap_{i}.png")

        # Save the data image
        save_image(data[i].cpu(), data_filename)

        # Save the heatmap
        plt.imshow(relevance[i], cmap='hot')
        plt.colorbar()
        plt.title(f"Difference: {difference[i]:.4f}_True_Label:{true_label[i][0]}")
        plt.axis('off')
        plt.savefig(heatmap_filename, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory
