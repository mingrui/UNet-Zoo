from __future__ import print_function

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import torchvision.transforms as tr

from data import BraTSDatasetUnet, BraTSDatasetLSTM
from losses import DICELossMultiClass
from models import UNet
from tqdm import tqdm
import numpy as np

DATA_FOLDER = '/mnt/960EVO/datasets/tiantan/2017-11/tiantan_preprocessed_png/'

dset_train = BraTSDatasetUnet(DATA_FOLDER, train=True,
                              keywords=['t2'],
                              im_size=[128, 128],
                              transform=tr.ToTensor())

train_loader = DataLoader(dset_train,
                          batch_size=64,
                          shuffle=True,
                          num_workers=1)

print('Train Dataset : ', len(train_loader.dataset))

model = UNet()

model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.99)

criterion = DICELossMultiClass()

