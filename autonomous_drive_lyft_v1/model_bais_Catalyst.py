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
    def forward(self, x): #torch.size([12, 5, 224, 224])
        x = self.backbone.encoder.conv1(x) #x.size() : torch.size([12, 64, 112, 112])  
        #(conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        x = self.backbone.encoder.bn1(x)       #x.size() : torch.size([12, 64, 112, 112]) 
        x = self.backbone.encoder.relu(x) #x.size() : torch.size([12, 64, 112, 112])
        x = self.backbone.encoder.maxpool(x)   #x.size() : torch.size([12, 64, 56, 56])  
        
        
        x = self.backbone.encoder.layer1(x)#x.size() : torch.size([12, 256, 56, 56])
    #      (layer1): Sequential(
    #   (0): Bottleneck(
    #     (conv1): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
    #     (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (relu): ReLU(inplace=True)
    
    #     (downsample): Sequential(
    #       (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #       (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     )
    #   )
    #   (1): Bottleneck(
    #     (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
    #     (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (relu): ReLU(inplace=True)
    #   )
    #   (2): Bottleneck(
    #     (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
    #     (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (conv3): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #     (relu): ReLU(inplace=True)
    #   )
    # )
       
        x = self.backbone.encoder.layer2(x)#x.size() : torch.Size([12, 512, 28, 28])
        x = self.backbone.encoder.layer3(x)#x.size() : torch.Size([12, 1024, 14, 14])
        x = self.backbone.encoder.layer4(x)  #x.size() : torch.Size([12, 2048, 7, 7])     
        
        
        x = self.backbone.decoder.p5(x)#x.size() : torch.Size([12, 256, 7, 7])
          # self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
          
        x = self.backbone.decoder.seg_blocks[0](x)#x.size() : torch.Size([12, 128, 56, 56])
        
        
        x = self.backbone.decoder.merge(x)#x.size() : torch.Size([128, 56, 56])
        
        x = self.backbone.segmentation_head(x)#x.size() : torch.Size([128, 1, 27])
        
        x = self.backbone.encoder.maxpool(x)#x.size() : torch.Size([128, 1, 14])
        x = torch.flatten(x, 1)#x.size() : torch.Size([128, 14])
        x = self.head(x)#x.size() : torch.Size([128, 4096])
        x = self.logit(x)   #x.size() : torch.Size([128, 100])
        x = x.permute(1, 0) #纬度交换#x.size() : torch.Size([100, 128])
        x = self.logit_final(x) #torch.Size([100, 12])
        return x
    
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LyftModel(cfg2)
model.to(device)
train_zarr = ChunkedDataset(dm.require(train_cfg['key'])).open()
train_dataset = AgentDataset(cfg2, train_zarr, rasterizer)

del train_cfg['key']
subset = torch.utils.data.Subset(train_dataset, range(0, 200))


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
        #torch.Size([12, 50, 1])
        # so what is target_availabilities ????
        criterion = torch.nn.MSELoss(reduction="none")
        loss = criterion(y_hat, y)
        loss = loss * target_availabilities
        loss = loss.mean()
        self.batch_metrics.update(
            {"loss": loss}
        )
        self.loader_metrics.update({
            "loss": loss}
        )
    # def training_epoch_end(self, outs):
    #     # log epoch metric
    #     # 方法：accem4
    #     self.log('accem4', self.accuracy.compute())
    
        
        
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
#trainer = Trainer(max_epochs=10, logger=neptune_logger)

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
       logdir="/Users/h/Downloads/lyft-motion-prediction-autonomous-vehicles/working",
      
       num_epochs=1,
       verbose=True,
       load_best_on_end=True,
       callbacks=[BatchOverfitCallback(train=10, valid=0.5), 
                 EarlyStoppingCallback(
           patience=2,
           metric_key="loss",
           loader_key="valid",
           minimize=True,
           
       )],
       
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
        
        
