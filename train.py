#!/usr/bin/env python
import random
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
import wandb
import os

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger 
from utils.utils import load_yaml, LRP
from model import LitModel, LitDataModule
from model_SSL import LitModelSSL, LitDataModuleSSL
from utils.utils import weight_decay_filter

def parse_arg():
    parser = argparse.ArgumentParser(description='Training or Testing the Model')
    parser.add_argument('--yaml', type=str, default='hyper.yaml', help="name of name.yaml in utils/yaml/ which should be used")
    parser.add_argument('--mode', type=str, default='train', help="default 'train', 'test' for testing (or 'lrp' and in test 'test+'); please provide a model path in yaml")
    return parser.parse_args()
         
def settings(hyperparameters): 

    filename = (
        f"{hyperparameters['dataset']}-"
        f"{hyperparameters['mode']}-"
        f"{hyperparameters['augmentation']}-"
        f"{'LARS' if hyperparameters['LARS'] else 'ADAMW'}-"
        f"{hyperparameters['loss']}-"
        f"{hyperparameters['kind'] if hyperparameters['pretrained'] else 'scratch'}-"
    )

    if hyperparameters['use_wandb']: 
        wandb.login(key=os.getenv('WANDB_API_KEY')) 
        wandb.init(project="DataAugmentationsPCAM", name=f"Run_{filename}", mode='online', entity="master_david")
        wandb_logger = WandbLogger(
                project="DataAugmentationsPCAM",
                entity="master_david")          

        # Log hyperparameters
        wandb_logger.log_hyperparams({
            "base_lr": hyperparameters['base_lr'],
            "batch_size": hyperparameters['batch_size'],
            "learning_rate": hyperparameters['learning_rate'],
            "LARS": hyperparameters['LARS'],
            "min_lr": hyperparameters['min_lr'],
            "num_epochs": hyperparameters['num_epochs'],
            "weight_decay": hyperparameters['weight_decay'],
            "momentum_start": hyperparameters['momentum_start'],
            "warmup_epochs": hyperparameters['warmup_epochs'],
            "model": hyperparameters['model'],
            "augmentation": hyperparameters['augmentation'],
            "pretrained": hyperparameters['pretrained'],
            "checkpoint_path": hyperparameters['checkpoint_path'],
            "num_classes": hyperparameters['num_classes'],
            "dataset": hyperparameters['dataset']
            })
        
        return wandb_logger, filename
    return None, filename

def main():

#### File Setup / ParaMeters     
    args = parse_arg()
    hyperparameters = load_yaml(args.yaml)
    seed = hyperparameters["seed"]
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision('medium')

   
    hyperparameters["learning_rate"] = hyperparameters["base_lr"] * hyperparameters["batch_size"] / 256
    wandb_logger, filename  = settings(hyperparameters) 

    if hyperparameters["mode"] == "SL":
        data_module = LitDataModule(hyperparameters)
    else: 
        data_module = LitDataModuleSSL(hyperparameters)

    data_module.setup() 
    train_data_loader = data_module.train_dataloader()
    data_loader_len = len(train_data_loader)
    hyperparameters['data_loader_len'] = data_loader_len

#### Model Loading
    if hyperparameters['checkpoint_path'] is not None:
        if hyperparameters["mode"] == "SL":
            model = LitModel.load_from_checkpoint(checkpoint_path=hyperparameters['checkpoint_path'], hyperparameters=hyperparameters)
        else:
            model = LitModelSSL.load_from_checkpoint(checkpoint_path=hyperparameters['checkpoint_path'], hyperparameters=hyperparameters)

    else:
        if hyperparameters["mode"] == "SL":
            model = LitModel(hyperparameters)
        else: 
            model = LitModelSSL(hyperparameters)

#### PyTorch Lighnting Configs
    checkpoint_callback = ModelCheckpoint(
    dirpath='./model_checkpoints/',  # Path where to save models
    filename=f'{filename}{{epoch}}-{{step}}',  # Filename of the checkpoints
    every_n_epochs=10,  # Save a checkpoint every 10 epochs
    save_on_train_epoch_end=True  # Save the checkpoint at the end of the training epoch
    )

    if 'accumulation_steps' in hyperparameters: 
        accumulation_steps = hyperparameters['accumulation_steps']

    else: 
        accumulation_steps = 1

    trainer = pl.Trainer(
    max_epochs=hyperparameters['num_epochs'],
    deterministic=True,
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=hyperparameters['accumulation_steps'],
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu',
    logger=wandb_logger,  #
    precision=32,
    )
#### Depending on args passed full-training, only test, only LRP or test + LRP
#### I removed the LRP code for now, so you will not be able to run it this time

    if args.mode == 'test': 
        trainer.test(model, data_module)

    elif args.mode == 'test+': 
        trainer.test(model, data_module)
        optimizer_name = 'LARS' if hyperparameters["LARS"] else 'AdamW'
        sub_dir_name = f"Loss_{hyperparameters["loss"]}_Opt_{optimizer_name}_Batch_{hyperparameters["batch_size"]}"
        save_path = './LRP/'+sub_dir_name+'/prediction_differences.pkl'
        LRP(model, save_path, sub_dir_name)

    elif hyperparameters['pretrained'] and hyperparameters['checkpoint_path'] and args.mode == 'lrp': 
        optimizer_name = 'LARS' if hyperparameters["LARS"] else 'AdamW'
        sub_dir_name = f"Loss_{hyperparameters["loss"]}_Opt_{optimizer_name}_Batch_{hyperparameters["batch_size"]}"
        save_path = './LRP/'+sub_dir_name+'/prediction_differences.pkl'
        LRP(model, save_path, sub_dir_name)

    else: 
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
        optimizer_name = 'LARS' if hyperparameters["LARS"] else 'AdamW'
        sub_dir_name = f"Loss_{hyperparameters["loss"]}_Opt_{optimizer_name}_Batch_{hyperparameters["batch_size"]}"
        save_path = './LRP/'+sub_dir_name+'/prediction_differences.pkl'
        LRP(model, save_path, sub_dir_name)

    print(f"Experiment ended, view results @ https://wandb.ai/master_david/DataAugmentationsPCAM")

if __name__ == "__main__":
    main()

   


