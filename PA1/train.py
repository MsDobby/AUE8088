"""
    [AUE8088] PA1: Image Classification
        - To run: (aue8088) $ python train.py
        - For better flexibility, consider using LightningCLI in PyTorch Lightning
"""
# PyTorch & Pytorch Lightning
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning import Trainer
import torch

# Custom packages
from src.dataset import TinyImageNetDatasetModule
from src.network import SimpleClassifier
import src.config as cfg

import argparse

torch.set_float32_matmul_precision('medium')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='setup')
    parser.add_argument('--wandb',action='store_true')
    parser.add_argument('--optimizer_params', type=str, default=None, help='Optimizer parameters as a dictionary string')
    parser.add_argument('--scheduler_params', type=str, default=None, help='Scheduler parameters as a dictionary string')
    args = parser.parse_args()

    optimizer_params = eval(args.optimizer_params) if args.optimizer_params else cfg.OPTIMIZER_PARAMS
    scheduler_params = eval(args.scheduler_params) if args.scheduler_params else cfg.SCHEDULER_PARAMS

    
    model = SimpleClassifier(
        model_name = cfg.MODEL_NAME,
        num_classes = cfg.NUM_CLASSES,
        optimizer_params = optimizer_params,
        scheduler_params = scheduler_params,
    )

    datamodule = TinyImageNetDatasetModule(
        batch_size = cfg.BATCH_SIZE,
    )

    wandb_logger = WandbLogger(
        project = cfg.WANDB_PROJECT,
        save_dir = cfg.WANDB_SAVE_DIR,
        entity = cfg.WANDB_ENTITY,
        name = cfg.WANDB_NAME,
        log_model = False
    )

    logger = wandb_logger if args.wandb else None
    
    trainer = Trainer(
        # num_sanity_val_steps=0,
        accelerator = cfg.ACCELERATOR,
        devices = cfg.DEVICES,
        precision = cfg.PRECISION_STR,
        max_epochs = cfg.NUM_EPOCHS,
        check_val_every_n_epoch = cfg.VAL_EVERY_N_EPOCH,
        logger = logger,
        callbacks = [
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(save_top_k=1, monitor='accuracy/val', mode='max'),
        ],
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.validate(ckpt_path='best', datamodule=datamodule)
