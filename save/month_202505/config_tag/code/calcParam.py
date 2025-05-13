import torch

import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import h5py

import datasets
import models
import utils
import numpy as np

with open('./configs/train-serenet/serenet_Nnum13_bubtub.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print('config loaded.')

model = models.make(config['model'])

params = utils.compute_num_params(model, text=False)

