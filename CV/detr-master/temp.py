#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:26:55 2021

@author: h
"""





import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

c=nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False);
print(c.weight.shape)
print(c.weight)
print(c)
