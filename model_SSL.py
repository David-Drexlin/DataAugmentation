import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from torchvision import models
import psutil
import os 
import heapq
import pickle
from torch.utils.data import DataLoader
from utils.augmentations import load_augmentations, normalization
from utils.utils import cosine_scheduler, weight_decay_filter, load_model, LARS
from utils.loss import loss_function  
from utils.load_data import HDF5Dataset, HDF5Dataset_Labels, MergedDataset
from sklearn.cluster import KMeans
import numpy as np 
import random 

#torch.set_float32_matmul_precision('medium') 

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=2048):
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)

class LitModelSSL(pl.LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.learning_rate = hyperparameters['learning_rate']
        self.accuracy = torchmetrics.Accuracy(task='binary')
        self.ssl_phase = True  # Default phase
        self.cosine_decay_lr_momentum = hyperparameters["cosine_decay_lr_momentum"]
        self.LARS = hyperparameters["LARS"]
        self.prediction_differences = []
        self.min_lr = self.hyperparameters['min_lr']
        self.epochs = self.hyperparameters['num_epochs']
        self.warmup_epochs = self.hyperparameters['warmup_epochs']
        self.data_loader_len= self.hyperparameters['data_loader_len']
        self.batch_size = hyperparameters["batch_size"]
        self.momentum_start = self.hyperparameters['momentum_start']
        self.weight_decay  = self.hyperparameters['weight_decay']
        self.accumulation_steps = self.hyperparameters['accumulation_steps']

        # Base encoder
        self.encoder = load_model(self.hyperparameters['model'], self.hyperparameters['num_classes'], self.hyperparameters["pretrained"])
        self.encoder.fc = nn.Identity()  # Remove the final fully connected layer
        self.classifier = nn.Linear(in_features=512, out_features=1)
        self.validation_step_outputs = []

        # Projection head
        self.projection_head = ProjectionHead(input_dim=512)  # ResNet18 outputs 512 features
        self.loss = loss_function(self.hyperparameters['loss'])(lambd=hyperparameters['lambda'])

    def forward(self, x1, x2=None):
        # Pass both inputs through the encoder and the projection head
        if x2 is not None: 
            d1 = self.encoder(x1)
            d2 = self.encoder(x2)
            
            z1 = self.projection_head(d1)
            z2 = self.projection_head(d2)
            return z1, z2, (d1, d2)
        else: 
            return self.projection_head(self.encoder(x1)), self.encoder(x1)
        
    def switch_to_classifier(self):
            self.ssl_phase = False
            self.loss = loss_function(self.hyperparameters['loss'], self.ssl_phase)

            # Freeze the parameters of base network and projector
            for param in self.base_network.parameters():
                param.requires_grad = False
            for param in self.projector.parameters():
                param.requires_grad = False
            for param in self.linear_classifier.parameters():
                param.requires_grad = True

    def training_step(self, batch, batch_idx):
        if self.ssl_phase:
            x1, x2, _ = batch  # Assuming your dataloader returns a tuple of (view1, view2)
            z1, z2, _ = self.forward(x1, x2)
            loss = self.loss(z1, z2) / self.accumulation_steps
            self.log('train_loss', loss)
        else: 
            x, y = batch
            _, d = self.forward(x)
            y_hat = self.classifier(d)
            loss = self.loss(y_hat.squeeze(), y.squeeze().float()) / self.accumulation_steps
        # Log training loss
            preds = (torch.sigmoid(y_hat) > 0.5).detach()
            self.log('linear_acc', self.accuracy(preds.squeeze(), y.squeeze().int()).detach())
            self.log('linea_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        if not self.ssl_phase:
            x, y, _ = batch
            _, d = self.forward(x)
            y_hat = self.classifier(d)
            loss = self.loss(y_hat.squeeze(), y.squeeze().float())

            # Compute and log test accuracy
            preds = torch.sigmoid(y_hat) > 0.5
            self.log('test_loss', loss, on_epoch=True, prog_bar=True)
            self.log('test_acc', self.accuracy(preds.squeeze(), y.squeeze().int()), on_epoch=True, prog_bar=True)

            probs = torch.sigmoid(y_hat).squeeze()

            # Store prediction differences and corresponding data
            prediction_differences = torch.abs(probs.squeeze() - y.squeeze().float()).detach().cpu().numpy()

            self.memory = psutil.virtual_memory().percent

            for i in range(len(prediction_differences)):
                # Ensure that prediction_differences[i] is a scalar

                difference = prediction_differences[i].item()
                heapq.heappush(self.prediction_differences, (difference, (batch_idx, i), x[i].clone(), y[i].cpu()))
                # Keep only top 5
                if len(self.prediction_differences) > 5:
                    heapq.heappop(self.prediction_differences)    
        else: 
            pass
    
    def on_test_epoch_end(self):
        if not self.ssl_phase:
            optimizer_name = 'LARS' if self.LARS else 'AdamW'
            sub_dir_name = f"Loss_{self.loss}_Opt_{optimizer_name}_Batch_{self.batch_size}"
            sub_dir_path = os.path.join('./LRP/', sub_dir_name)

            os.makedirs(sub_dir_path, exist_ok=True)

            save_path = os.path.join(sub_dir_path, 'prediction_differences.pkl')

            with open(save_path, 'wb') as f:
                pickle.dump(self.prediction_differences, f)
        else: 
            pass  
    
    def validation_step(self, batch, batch_idx):
        if self.ssl_phase:
            x, y = batch  # Assuming your dataloader returns a tuple of (view1, view2)
            _, z = self.forward(x)
            self.validation_step_outputs.append({'features': z, 'labels': y})
        else:
            pass

    def on_validation_epoch_end(self):
        if self.ssl_phase:
            all_features = torch.cat([o['features'] for o in self.validation_step_outputs], dim=0)
            all_labels = torch.cat([o['labels'] for o in self.validation_step_outputs], dim=0)

            # Train K-means on the features
            kmeans = KMeans(n_clusters=2, random_state=42).fit(all_features.cpu().numpy())

            # Apply K-means clustering
            preds = torch.tensor(kmeans.labels_, dtype=torch.int32, device=all_labels.device)

            self.log('test_acc', self.accuracy(preds.squeeze(), all_labels.squeeze()), on_epoch=True, prog_bar=True)
            self.validation_step_outputs.clear()  
        else:
            pass
        
    def configure_optimizers(self):
        
        if self.LARS:
            combined_params = list(self.encoder.parameters()) + list(self.projection_head.parameters())
            optimizer = LARS(combined_params, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9, eta=0.001, weight_decay_filter=weight_decay_filter)
        else: 
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        if self.cosine_decay_lr_momentum:
            
            # Assuming cosine_scheduler returns a PyTorch Learning Rate Scheduler
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

class LitDataModuleSSL(pl.LightningDataModule):
    def __init__(self, hyperparameters):
        super().__init__()
        augmentation_mapping = load_augmentations()

        if hyperparameters["augmentation"][0] in augmentation_mapping:
            augmentation1 = augmentation_mapping[hyperparameters["augmentation"][0]]

        if hyperparameters["augmentation"][1] in augmentation_mapping:
            augmentation2 = augmentation_mapping[hyperparameters["augmentation"][1]]
                    
        self.batch_size = hyperparameters['batch_size']
        self.dataset = hyperparameters['dataset']
        self.transform1 = augmentation1
        self.transform2 = augmentation2

    def setup(self, stage=None):
        # Split data and set up for train, val, test

        #if self.dataset is 'pcam':
        x1 = HDF5Dataset('./data/pcam/camelyonpatch_level_2_split_train_x.h5-002', 'x', self.transform1)
        x2 = HDF5Dataset('./data/pcam/camelyonpatch_level_2_split_train_x.h5-002', 'x', self.transform2)
        y =  HDF5Dataset_Labels('./data/pcam/camelyonpatch_level_2_split_train_y.h5', 'y')
        self.C16_train = MergedDataset((x1, x2),y)

        x = HDF5Dataset('./data/pcam/camelyonpatch_level_2_split_valid_x.h5', 'x', normalization(), train=False)
        y = HDF5Dataset_Labels('./data/pcam/camelyonpatch_level_2_split_valid_y.h5', 'y') 
        self.C16_val = MergedDataset(x,y)

        x = HDF5Dataset('./data/pcam/camelyonpatch_level_2_split_test_x.h5', 'x', normalization(), train=False)
        y =  HDF5Dataset_Labels('./data/pcam/camelyonpatch_level_2_split_test_y.h5', 'y') 
        self.C16_test = MergedDataset(x,y,is_test=True)


    def train_dataloader(self):
        return DataLoader(self.C16_train, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.C16_val, batch_size=self.batch_size, shuffle=False, num_workers=4,persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.C16_test, batch_size=self.batch_size, shuffle=False, num_workers=7)
