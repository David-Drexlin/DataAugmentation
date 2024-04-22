import torchmetrics
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from utils.load_data import HDF5Dataset, HDF5Dataset_Labels, MergedDataset
from utils.utils import cosine_scheduler, weight_decay_filter, load_model, LARS, load_path
from utils.loss import loss_function
from utils.augmentations import load_augmentations
import pytorch_lightning as pl
import torchvision
import psutil
import heapq
import os
import pickle

class LitModel(pl.LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.transform = self.hyperparameters['augmentation']
        self.learning_rate = self.hyperparameters['learning_rate']
        
        self.accuracy = torchmetrics.Accuracy(task='multiclass',num_classes= self.hyperparameters['num_classes'])
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes= self.hyperparameters['num_classes'])
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass',num_classes= self.hyperparameters['num_classes'])
                
        self.model = load_model(self.hyperparameters['model'], self.hyperparameters['num_classes'], self.hyperparameters["pretrained"], self.hyperparameters["kind"])
        self.LARS = self.hyperparameters['LARS']
        self.cosine_decay_lr_momentum = self.hyperparameters['cosine_decay_lr_momentum']
        self.momentum_start = self.hyperparameters['momentum_start']
        self.weight_decay  = self.hyperparameters['weight_decay']
        self.loss = loss_function(self.hyperparameters['loss'])
        self.min_lr = self.hyperparameters['min_lr']
        self.epochs = self.hyperparameters['num_epochs']
        self.warmup_epochs = self.hyperparameters['warmup_epochs']
        self.data_loader_len= self.hyperparameters['data_loader_len']
        self.prediction_differences = [] ## used for LRP to store references during test #currently deactivated due to clean-up

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch    

        y_hat = self.forward(x)
        loss = self.loss(y_hat.float(), y) # CrossEntropy
        preds = torch.argmax(y_hat, dim=1)
    
        self.log('accuracy', self.accuracy(preds, y), on_epoch=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch   

        y_hat = self.forward(x)
        loss = self.loss(y_hat.float(), y)
        preds = torch.argmax(y_hat, dim=1)

        self.log('val_accuracy', self.val_accuracy(preds, y), on_epoch=True)
        self.log('val_loss', loss, on_epoch=True)
        
    def test_step(self, batch, batch_idx): 
        x, y = batch    

        y_hat = self.forward(x)
        loss = self.loss(y_hat.float(), y)
        preds = torch.argmax(y_hat, dim=1)

        self.log('test_acc', self.test_accuracy(preds, y), on_epoch=True)
        self.log('test_loss', loss, on_epoch=True)

    def configure_optimizers(self):
        
        if self.LARS: 
            optimizer = LARS(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9, eta=0.001, weight_decay_filter=weight_decay_filter)
        else: 
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        if self.cosine_decay_lr_momentum:
            
            lr_scheduler = cosine_scheduler(
                self.learning_rate,
                self.min_lr,
                self.epochs, self.data_loader_len,
                warmup_epochs=self.warmup_epochs
            )
            
            momentum_scheduler = cosine_scheduler(
                self.momentum_start,
                1,
                self.epochs, self.data_loader_len
            )

            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'name': 'lr_scheduler',
            }

            momentum_scheduler_config = {
                'scheduler': momentum_scheduler,
                'interval': 'step',
                'name': 'momentum_scheduler',
            }

            return {"optimizer": optimizer, "lr_schedulers": [lr_scheduler_config, momentum_scheduler_config]}
                
        return optimizer
    
class LitDataModule(pl.LightningDataModule):
    def __init__(self, hyperparameters):
        super().__init__()
        self.dataset = hyperparameters['dataset']
        
        if self.dataset == "crc": 
            # handling for h5 file configuration
            self.transform = load_augmentations(hyperparameters["augmentation"], True)
            self.val_transform = load_augmentations(hyperparameters["augmentation_val"], True)
            self.test_transform = load_augmentations(hyperparameters["augmentation_test"], True)
            
            paths = load_path(hyperparameters["dataset"])
            self.train_x_path = paths["train_data_file"]
            self.train_y_path = paths["train_label_file"]
            self.val_x_path = paths["valid_data_file"]
            self.val_y_path = paths["valid_label_file"]
            self.test_x_path = paths["test_data_file"]
            self.test_y_path = paths["test_label_file"]

        else: 
            self.transform = load_augmentations(hyperparameters["augmentation"])
            self.val_transform = load_augmentations(hyperparameters["augmentation_val"])
            self.test_transform = load_augmentations(hyperparameters["augmentation_test"])

        self.dataset = hyperparameters["dataset"]
        self.batch_size = hyperparameters['batch_size']
    
    def setup(self, stage=None):
        if self.dataset == "CRC-NoNorm": 
            self.C16_train = torchvision.datasets.ImageFolder('./data/CRC/NCT-CRC-HE-100K-NONORM', self.transform)
            self.C16_val = torchvision.datasets.ImageFolder('./data/CRC/CRC-VAL-HE-7K', self.test_transform)

        elif self.dataset == "CRC": 
            self.C16_train = torchvision.datasets.ImageFolder('./data/CRC/NCT-CRC-HE-100K', self.transform)
            self.C16_val = torchvision.datasets.ImageFolder('./data/CRC/CRC-VAL-HE-7K', self.test_transform)

        else: 
            x = HDF5Dataset(self.train_x_path, self.transform)
            y = HDF5Dataset_Labels(self.train_y_path) 
            self.C16_train = MergedDataset(x,y)

            x = HDF5Dataset(self.val_x_path, self.val_transform, train=False)
            y = HDF5Dataset_Labels(self.val_y_path) 
            self.C16_val = MergedDataset(x,y)

            x = HDF5Dataset(self.test_x_path, self.test_transform, train=False)
            y =  HDF5Dataset_Labels(self.test_y_path) 
            self.C16_test = MergedDataset(x,y)

    def test_dataloader(self):
        return DataLoader(self.C16_test, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def train_dataloader(self):
        return DataLoader(self.C16_train, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.C16_val, batch_size=self.batch_size, shuffle=False, num_workers=4,persistent_workers=True)

