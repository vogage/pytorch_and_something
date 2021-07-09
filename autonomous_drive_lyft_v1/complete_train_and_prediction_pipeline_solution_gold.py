#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:11:56 2021

@author: h
"""

from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
from tqdm import tqdm


import l5kit
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import matplotlib.pyplot as plt


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import time

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed(42)





# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'data_path': "/Users/h/Downloads/lyft-motion-prediction-autonomous-vehicles",
    'model_params': {
        'model_architecture': 'resnet34',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "model_resnet34_output",
        'lr': 1e-3,
        # 'weight_path': "/Users/h/Downloads/lyft-motion-prediction-autonomous-vehicles/input/lyft-pretrained-model-hv/model_multi_update_lyft_public.pth",
        'weight_path': "",
        'train': True,
        'predict': False,
        'render_ego_history':True,
        'step_time': 0.1
    },

    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5,
        'set_origin_to_bottom': True
    },

    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 0
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 0
    },

    'train_params': {
        'max_num_steps': 101,
        'checkpoint_every_n_steps': 20,
    }
}

# set env variable for data
DIR_INPUT = cfg["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)


# ===== INIT TRAIN DATASET============================================================
train_cfg = cfg["train_data_loader"]

rasterizer = build_rasterizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
print("==================================TRAIN DATA==================================")
print(train_dataset)




#====== INIT TEST DATASET=============================================================
test_cfg = cfg["test_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
test_dataloader = DataLoader(test_dataset,shuffle=test_cfg["shuffle"],batch_size=test_cfg["batch_size"],
                             num_workers=test_cfg["num_workers"])
print("==================================TEST DATA==================================")
print(test_dataset)



def visualize_trajectory(dataset, index, title="target_positions movement with draw_trajectory"):
    data = dataset[index]
    im = data["image"].transpose(1, 2, 0)
    # 共25个channels，其中11个chanel作为agent，11个channel作为ego，剩下的3个作为image
    
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, radius=1, yaws=data["target_yaws"])

    plt.title(title)
    plt.imshow(im[::-1])
    plt.show()
    
    
    
plt.figure(figsize = (8,6))
visualize_trajectory(train_dataset, index=90)



# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
import numpy as np

import torch
from torch import Tensor


def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)




class LyftMultiModel(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        architecture = cfg["model_params"]["model_architecture"]
        #  'model_architecture': 'resnet34',
        backbone = eval(architecture)(pretrained=True, progress=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        #  num_history_channels = 22
        #  'history_num_frames': 10,
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        
        if architecture == "resnet50":
            backbone_out_features = 2048
        else:
            backbone_out_features = 512

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        # 'future_num_frames': 50,
        
        num_targets = 2 * self.future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)
        # out_features = 303

    def forward(self, x): #torch.Size([2, 25, 224, 224])
        x = self.backbone.conv1(x)#torch.Size([2, 64, 112, 112])
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        # MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        x = self.backbone.maxpool(x) #torch.Size([2, 64, 56, 56])

        x = self.backbone.layer1(x) #torch.Size([2, 64, 56, 56])
        x = self.backbone.layer2(x) #torch.Size([2, 128, 28, 28])
        x = self.backbone.layer3(x) #torch.Size([2, 256, 14, 14])
        x = self.backbone.layer4(x) #torch.Size([2, 512, 7, 7])

        x = self.backbone.avgpool(x) #torch.Size([2, 512, 1, 1])
        x = torch.flatten(x, 1) #torch.Size([2, 512])

        x = self.head(x) #torch.Size([2, 4096])
        x = self.logit(x) #torch.Size([2, 303])

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape # x.shape=torch.Size([2, 303])
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        #pred.size() = torch.Size([2, 300]) 
        #confidences.size()= torch.size([2,3])
        
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        #pred.size() = torch.Size([2, 3, 50, 2])
        
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences
    
    
def forward(data, model, device, criterion = pytorch_neg_multi_log_likelihood_batch):
    inputs = data["image"].to(device)
    # print(inputs.size())
    # torch.Size([2, 25, 224, 224])
    
    target_availabilities = data["target_availabilities"].to(device)
    
    targets = data["target_positions"].to(device)
    
    # Forward pass
    preds, confidences = model(inputs)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences    

# ==== INIT MODEL=================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LyftMultiModel(cfg)

#load weight if there is a pretrained model
weight_path = cfg["model_params"]["weight_path"]
if weight_path:
    model.load_state_dict(torch.load(weight_path))

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
print(f'device {device}')


# ==== TRAINING LOOP =========================================================
if cfg["model_params"]["train"]:
    
    tr_it = iter(train_dataloader)
    
    progress_bar = tqdm(range(cfg["train_params"]["max_num_steps"]))
    # show a progress meter
    """
    Decorate an iterable object, returning an iterator which acts exactly
    like the original iterable, but prints a dynamically updating
    progressbar every time a value is requested.
    """
    
    num_iter = cfg["train_params"]["max_num_steps"]
    losses_train = []
    iterations = []
    metrics = []
    times = []
    model_name = cfg["model_params"]["model_name"]
    start = time.time()
    for i in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)
            
        # 处理异常，比如缺页、除0、非法寻址
        model.train()
        torch.set_grad_enabled(True)
        
        loss, _, _ = forward(data, model, device)

        # Backward pass
        optimizer.zero_grad()
        # It is beneficial to zero out gradients when building a neural network.
        # This is because by default, gradients are accumulated in buffers
        # (i.e, not overwritten) whenever .backward() is called.
        
        loss.backward()
        
        optimizer.step()
        

        losses_train.append(loss.item())

        progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
        if i % cfg['train_params']['checkpoint_every_n_steps'] == 0:
            torch.save(model.state_dict(), f'{model_name}_{i}.pth')
            iterations.append(i)
            metrics.append(np.mean(losses_train))
            times.append((time.time()-start)/60)

    results = pd.DataFrame({'iterations': iterations, 'metrics (avg)': metrics, 'elapsed_time (mins)': times})
    results.to_csv(f"train_metrics_{model_name}_{num_iter}.csv", index = False)
    print(f"Total training time is {(time.time()-start)/60} mins")
    print(results.head())
    
    
    
    
    
# ==== EVAL LOOP ================================================================
if cfg["model_params"]["predict"]:
    
    model.eval()
    torch.set_grad_enabled(False)

    # store information for evaluation
    future_coords_offsets_pd = []
    timestamps = []
    confidences_list = []
    agent_ids = []

    progress_bar = tqdm(test_dataloader)
    """
    Decorate an iterable object, returning an iterator which acts exactly
    like the original iterable, but prints a dynamically updating
    progressbar every time a value is requested.
    """
    
    for data in progress_bar:
        
        _, preds, confidences = forward(data, model, device)
    
        #fix for the new environment
        preds = preds.cpu().numpy() # don't know why here use cpu()
        world_from_agents = data["world_from_agent"].numpy()
        centroids = data["centroid"].numpy()
        coords_offset = []
        
        # convert into world coordinates and compute offsets
        for idx in range(len(preds)):
            for mode in range(3):
                preds[idx, mode, :, :] = transform_points(preds[idx, mode, :, :], world_from_agents[idx]) - centroids[idx][:2]
                """
                Transform a set of 2D/3D points using the given transformation matrix.
                Assumes row major ordering of the input points. The transform function has 3 modes:
                - points (N, F), transf_matrix (F+1, F+1)
                    all points are transformed using the matrix and the output points have shape (N, F).
                - points (B, N, F), transf_matrix (F+1, F+1)
                    all sequences of points are transformed using the same matrix and the output points have shape (B, N, F).
                    transf_matrix is broadcasted.
                - points (B, N, F), transf_matrix (B, F+1, F+1)
                    each sequence of points is transformed using its own matrix and the output points have shape (B, N, F).
            
                Note this function assumes points.shape[-1] == matrix.shape[-1] - 1, which means that last
                rows in the matrices do not influence the final results.
                For 2D points only the first 2x3 parts of the matrices will be used.
            
                Args:
                    points (np.ndarray): Input points of shape (N, F) or (B, N, F)
                    with F = 2 or 3 depending on input points are 2D or 3D points.
                    transf_matrix (np.ndarray): Transformation matrix of shape (F+1, F+1) or (B, F+1, F+1) with F = 2 or 3.
            
                Returns:
                    np.ndarray: Transformed points of shape (N, F) or (B, N, F) depending on the dimensions of the input points.
                """
        future_coords_offsets_pd.append(preds.copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        timestamps.append(data["timestamp"].numpy().copy())
        agent_ids.append(data["track_id"].numpy().copy()) 
        
        #Tensor.copy_(src, non_blocking=False) → Tensor
        """
        Copies the elements from src into self tensor and returns self.
        The src tensor must be broadcastable with the self tensor. 
        It may be of a different data type or reside on a different device.
        
        Parameters:
        src (Tensor) – the source tensor to copy from
        non_blocking (bool) – if True and this copy is between CPU and GPU, 
        the copy may occur asynchronously with respect to the host. 
        For other cases, this argument has no effect.   
        
        """
#create submission to submit to Kaggle
pred_path = 'submission.csv'
write_pred_csv(pred_path,
           timestamps=np.concatenate(timestamps),
           track_ids=np.concatenate(agent_ids),
           coords=np.concatenate(future_coords_offsets_pd),
           confs = np.concatenate(confidences_list)
          )


