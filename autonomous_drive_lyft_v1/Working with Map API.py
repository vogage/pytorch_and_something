#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 14:48:56 2021

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
# set env variable for data
os.environ["L5KIT_DATA_FOLDER"] = "/Users/h/Downloads/lyft-motion-prediction-autonomous-vehicles"


# get config
cfg = load_config_data("/Users/h/Downloads/lyft-motion-prediction-autonomous-vehicles/visualisation_config2.yaml")
#https://github.com/lyft/l5kit/blob/master/examples/visualisation/visualisation_config.yaml

from IPython.display import display, clear_output, HTML
from l5kit.data import ChunkedDataset, LocalDataManager
from l5kit.dataset import EgoDataset, AgentDataset

dm = LocalDataManager()
dataset_path = dm.require(cfg["val_data_loader"]["key"]);
rasterizer = build_rasterizer(cfg, dm)
zarr_dataset = ChunkedDataset(dataset_path)
train_dataset_a = AgentDataset(cfg, zarr_dataset, rasterizer)
zarr_dataset.open()
print(zarr_dataset)


def plot_image(map_type, ax, agent=False):
    cfg["raster_params"]["map_type"] = map_type
    rast = build_rasterizer(cfg, dm)
    if agent:
        dataset = AgentDataset(cfg, zarr_dataset, rast)
    else:
        dataset = EgoDataset(cfg, zarr_dataset, rast)
    scene_idx = 2
    indexes = dataset.get_scene_indices(scene_idx)
    images = []
    for idx in indexes:    
        data = dataset[idx]
        im = data["image"].transpose(1, 2, 0)
        im = dataset.rasterizer.to_rgb(im)
        target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
        center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
        draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])
        clear_output(wait=True)
        ax.imshow(im[::-1])
        
        
agents = zarr_dataset.agents
probabilities = agents["label_probabilities"]
labels_indexes = np.argmax(probabilities, axis=1)
counts = []
for idx_label, label in enumerate(PERCEPTION_LABELS):
    counts.append(np.sum(labels_indexes == idx_label))
    
table = PrettyTable(field_names=["label", "counts"])
for count, label in zip(counts, PERCEPTION_LABELS):
    table.add_row([label, count])
print(table)



#What if I want to visualise the Autonomous Vehicle (AV)?
dataset = EgoDataset(cfg, zarr_dataset, rasterizer)
data = dataset[80]

im = data["image"].transpose(1, 2, 0)
im = dataset.rasterizer.to_rgb(im)
target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR,1,data["target_yaws"])


plt.imshow(im[::-1])
plt.show()


#What if I want to change the rasterizer?
cfg["raster_params"]["map_type"] = "py_satellite"
rasterizer = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, zarr_dataset, rasterizer)
data = dataset[80]

im = data["image"].transpose(1, 2, 0)
im = dataset.rasterizer.to_rgb(im)
target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
draw_trajectory(im, target_positions_pixels,TARGET_POINTS_COLOR,1,data["target_yaws"])
#draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])

plt.imshow(im[::-1])
plt.show()


#What if I want to visualise an agent?¶
dataset = AgentDataset(cfg, zarr_dataset, rasterizer)
data = dataset[0]

im = data["image"].transpose(1, 2, 0)
im = dataset.rasterizer.to_rgb(im)
target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])

plt.imshow(im)
plt.show()


# what a entire scence look like ?
from IPython.display import display, clear_output
import PIL
 
#cfg["raster_params"]["map_type"] = "py_semantic"
cfg["raster_params"]["map_type"] = "py_satellite"
rast = build_rasterizer(cfg, dm)
dataset = EgoDataset(cfg, zarr_dataset, rast)
scene_idx = 1
indexes = dataset.get_scene_indices(scene_idx)
images = []

for idx in indexes:
    
    data = dataset[idx]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"], data["raster_from_agent"])
    center_in_pixels = np.asarray(cfg["raster_params"]["ego_center"]) * cfg["raster_params"]["raster_size"]
    draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, yaws=data["target_yaws"])
    clear_output(wait=True)
    display(PIL.Image.fromarray(im))


fig, axes = plt.subplots(1, 1, figsize=(15, 10))
for i, key in enumerate(["py_satellite", "py_semantic", 'stub_debug', "box_debug"]):
    plot_image(key, ax=axes, agent=True)
    
    
import pandas as pd
scenes = zarr_dataset.scenes
scenes_df = pd.DataFrame(scenes)
scenes_df.columns = ["data"]; features = ['frame_index_interval', 'host', 'start_time', 'end_time']
for i, feature in enumerate(features):
    scenes_df[feature] = scenes_df['data'].apply(lambda x: x[i])
scenes_df.drop(columns=["data"],inplace=True)
print(f"scenes dataset: {scenes_df.shape}")
scenes_df.head()


frms = pd.read_csv("/Users/h/Downloads/lyft-motion-prediction-autonomous-vehicles/frames_0_124167_124167.csv")
frms.head()


import seaborn as sns
colormap = plt.cm.magma
cont_feats = ["ego_rotation_xx", "ego_rotation_xy", "ego_rotation_xz", "ego_rotation_yx", "ego_rotation_yy", "ego_rotation_yz", "ego_rotation_zx", "ego_rotation_zy", "ego_rotation_zz"]
plt.figure(figsize=(16,12));
plt.title('Pearson correlation of features', y=1.05, size=15);
sns.heatmap(frms[cont_feats].corr(),linewidths=0.1,vmax=1.0, square=True, 
            cmap=colormap, linecolor='white', annot=True);




    

