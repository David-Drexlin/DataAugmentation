import torch
from torch import nn
import torchvision.models as models

def loss_function(loss, SSL=True): 
    loss_mapping = {
        "binary_cross_entropy_with_logits": torch.nn.functional.binary_cross_entropy_with_logits,
        "binary_cross_entropy" : torch.nn.functional.binary_cross_entropy,
        "barlow_twins": BarlowTwinsLoss, 
        "cross_entropy": torch.nn.functional.cross_entropy
    }
    
    loss_constructor = loss_mapping.get(loss)
    
    if not SSL: # use linear classifier when SSL training has concluded 
        loss_constructor = torch.nn.functional.binary_cross_entropy

    return loss_constructor

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd=0.0051):
        super().__init__()
        self.lambd = lambd
        self.bn = nn.BatchNorm1d(2048, affine=False)  # Assuming the output dim of your projection head is 8192

    def forward(self, z1, z2):
        z1 = self.bn(z1)
        z2 = self.bn(z2)
        c = torch.mm(z1.T, z2)
        c.div_(z1.size(0))

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m, "Matrix must be square."
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()