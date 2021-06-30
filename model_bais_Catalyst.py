#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:27:24 2021

@author: h
"""

import numpy as np
import l5kit, os
import matplotlib.pyplot as plt
from l5kit.rasterization import build_rasterizer
from l5kit.configs import load_config_data
from l5kit.visualization import draw_trajectory, TARGET_POINTS_COLOR
from l5kit.geometry import transform_points
from tqdm import tqdm
from collections import Counter
from l5kit.data import PERCEPTION_LABELS
from prettytable import PrettyTable
import omegaconf
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.utils as utils
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning.loggers import NeptuneLogger


#Catalyst
from catalyst import dl
from catalyst.dl import (
    SupervisedRunner, BatchOverfitCallback,EarlyStoppingCallback
)
#from catalyst.callbacks import EarlyStoppingCallback
from catalyst.loggers.wandb import WandbLogger
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/Users/h/Downloads/lyft-motion-prediction-autonomous-vehicles"


# get config
cfg = load_config_data("/Users/h/Downloads/lyft-motion-prediction-autonomous-vehicles/lyft_config_files/visualisation_config2.yaml")
#https://github.com/lyft/l5kit/blob/master/examples/visualisation/visualisation_config.yaml

from IPython.display import display, clear_output, HTML
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset



cfg2 = load_config_data("/Users/h/Downloads/lyft-motion-prediction-autonomous-vehicles/lyft_config_files/agent_motion_config.yaml")
cfg2 = omegaconf.OmegaConf.create(cfg2)
train_cfg = omegaconf.OmegaConf.to_container(cfg2.train_data_loader)
validation_cfg = omegaconf.OmegaConf.to_container(cfg2.val_data_loader)
# Rasterizer
dm = LocalDataManager()
rasterizer = build_rasterizer(cfg2, dm)


class LyftModel(torch.nn.Module):
    def __init__(self, cfg: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.backbone = smp.FPN(encoder_name="resnext50_32x4d", classes=1)
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        self.backbone.encoder.conv1 = nn.Conv2d(
            num_in_channels,
             self.backbone.encoder.conv1.out_channels,
            kernel_size= self.backbone.encoder.conv1.kernel_size,
            stride= self.backbone.encoder.conv1.stride,
            padding= self.backbone.encoder.conv1.padding,
            bias=False,
        ) 
        backbone_out_features = 14
        num_targets = 2 * cfg["model_params"]["future_num_frames"]
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=14, out_features=4096),
        )
        self.backbone.segmentation_head = nn.Sequential(nn.Conv1d(56, 1, kernel_size=3, stride=2), nn.Dropout(0.2), nn.ReLU())
        self.logit = nn.Linear(4096, out_features=num_targets)
        self.logit_final = nn.Linear(128, 12)
        self.num_preds = num_targets * 3
    def forward(self, x):
        x = self.backbone.encoder.conv1(x)
        x = self.backbone.encoder.bn1(x)        
        x = self.backbone.encoder.relu(x)
        x = self.backbone.encoder.maxpool(x)        
        x = self.backbone.encoder.layer1(x)
        x = self.backbone.encoder.layer2(x)
        x = self.backbone.encoder.layer3(x)
        x = self.backbone.encoder.layer4(x)        
        x = self.backbone.decoder.p5(x)
        x = self.backbone.decoder.seg_blocks[0](x)
        x = self.backbone.decoder.merge(x)
        x = self.backbone.segmentation_head(x)
        x = self.backbone.encoder.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        x = self.logit(x)   
        x = x.permute(1, 0)
        x = self.logit_final(x)
        return x
    
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LyftModel(cfg2)
model.to(device)
train_zarr = ChunkedDataset(dm.require(train_cfg['key'])).open()
train_dataset = AgentDataset(cfg2, train_zarr, rasterizer)

del train_cfg['key']
subset = torch.utils.data.Subset(train_dataset, range(0, 1100))


train_dataloader = DataLoader(subset,
                              **train_cfg)
val_zarr = ChunkedDataset(dm.require(validation_cfg['key'])).open()

del validation_cfg['key']

val_dataset = AgentDataset(cfg2, val_zarr, rasterizer)
subset = torch.utils.data.Subset(val_dataset, range(0, 50))
val_dataloader = DataLoader(subset,
                              **validation_cfg)

optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loaders = {
    "train": train_dataloader,
    "valid": val_dataloader
}



class LyftRunner(dl.SupervisedRunner):
    def predict_batch(self, batch):
        return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))
    def handle_batch(self, batch):
        x, y = batch['image'], batch['target_positions']
        y_hat = self.model(x).view(y.shape)
        target_availabilities = batch["target_availabilities"].unsqueeze(-1)
        criterion = torch.nn.MSELoss(reduction="none")
        loss = criterion(y_hat, y)
        loss = loss * target_availabilities
        loss = loss.mean()
        self.batch_metrics.update(
            {"loss": loss}
        )
    
        
        
#    %%time
#device = torch.device('cpu',0)
#  model: Torch model instance
#         engine: IEngine instance
#         input_key: key in ``runner.batch`` dict mapping for model input
#         output_key: key for ``runner.batch`` to store model output
#         target_key: key in ``runner.batch`` dict mapping for target
#         loss_key: key for ``runner.batch_metrics`` to store criterion loss out
runner = LyftRunner(input_key="image",output_key="logits")

#contain MONITORING:
# The Neptune logger can be used in the online mode or offline (silent) mode. 
# To log experiment data in online mode, NeptuneLogger requires an API key. 
# In offline mode, the logger does not connect to Neptune

# arguments made to NeptuneLogger are passed on to the neptune.experiments.Experiment class
neptune_logger = NeptuneLogger(
    offline_mode=True,
    project_name='USER_NAME/PROJECT_NAME',
    experiment_name='default',  # Optional,
    params={'max_epochs': 10},  # Optional,
    tags=['pytorch-lightning', 'catalyst'],
    order=0,# Optional,

)
# trainer = Trainer(max_epochs=10, logger=neptune_logger)

# runner.train(
#        model=model,
#        optimizer=optimizer,
#        loaders=loaders,
#        logdir="../working",
#        num_epochs=4,
#        verbose=True,
#        load_best_on_end=True,
#        callbacks=[BatchOverfitCallback(train=10, valid=0.5), 
#                  EarlyStoppingCallback(
#            patience=2,
#            metric_key="loss",
#            loader_key="valid",
#            minimize=True,
#            order=1),
#             WandbLogger(project="dertaismus",name= 'Example')
#                  ]
#    )
   




runner.train(
       model=model,
       optimizer=optimizer,
       loaders=loaders,
       logdir="../working",
       num_epochs=4,
       verbose=True,
       load_best_on_end=True,
       callbacks=[BatchOverfitCallback(train=10, valid=0.5), 
                 EarlyStoppingCallback(
           patience=2,
           metric_key="loss",
           loader_key="valid",
           minimize=True,
           
       )
       
                 ]
   )
#runner.train()
# patience – number of epochs with no improvement after which training will be stopped.

# loader_key – loader key for early stopping (based on metric score over the dataset)

# metric_key – metric key for early stopping (based on metric score over the dataset)

# minimize – if True then expected that metric should decrease and early stopping will 
# be performed only when metric stops decreasing. If False then expected that metric 
# should increase. Default value True.

# min_delta – minimum change in the monitored metric to qualify as an improvement, 
# i.e. an absolute change of less than min_delta, will count as no improvement, d
# efault value is 1e-6.




# train for more steps and loss will not be low
# or at least not as pathetic as the loss over here      
        
        
