#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 15:11:56 2021

@author: h
"""

import os,sys
sys.path.insert(0, "../input/best-submission/src")
import models.pointnet as pointnet
# from data.dataset import CustomLyftDataset_V5 as CustomLyftDataset
from training.trainer import get_last_checkpoint, BaseLightningModule
from training.configs.base import GenericConfig
from data.dataset import collate_V5 as collate, CustomLyftDataset_V7 as CustomLyftDataset
    
    
