# %% -*- coding: utf-8 -*-
'''
Author: Shreyas Padhy
Driver file for Unet and BDC-LSTM Implementation
'''

from __future__ import print_function

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as tr

from data import BraTSDatasetUnet, BraTSDatasetLSTM, UnetPred
from losses import DICELoss, DICELossMultiClass
from models import UNetSmall
from tqdm import tqdm
import scipy.io as sio
import numpy as np

from plot_ims import save_prediction, plot_test

# %% import transforms

# %% Training settings
parser = argparse.ArgumentParser(description='UNet+BDCLSTM for BraTS Dataset')
parser.add_argument('--batch-size', type=int, default=3, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=6, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--train', action='store_true', default=False,
                    help='Argument to train model (default: False)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training (default: False)')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='batches to wait before logging training status')
parser.add_argument('--size', type=int, default=512, metavar='N',
                    help='imsize')
parser.add_argument('--load', type=str, default=None, metavar='str',
                    help='weight file to load (default: None)')
parser.add_argument('--data-folder', type=str,
                    default='none',
                    metavar='str',
                    help='folder that contains data (default: test dataset)')
parser.add_argument('--save', type=str, default='OutMasks', metavar='str',
                    help='Identifier to save npy arrays with')
parser.add_argument('--modality', type=str, default='t2', metavar='str',
                    help='Modality to use for training (default: flair)')
parser.add_argument('--optimizer', type=str, default='SGD', metavar='str',
                    help='Optimizer (default: SGD)')
parser.add_argument('--clip', action='store_true', default=False,
                    help='enables gradnorm clip of 1.0 (default: False)')
parser.add_argument('--pred-input', type=str, default=None, metavar='str',
                    help='folder that contains data to make predctions')
parser.add_argument('--pred-output', type=str, default=None, metavar='str',
                    help='folder that contains data to make predctions')
parser.add_argument('--batch-out-folder', type=str, default=None, metavar='str',
                    help='folder that contains data to make predctions')
parser.add_argument('--channels', type=int, default=1, metavar='N',
                    help='number of channels, 1 for grayscale, 3 for rgb (default: 1)')
parser.add_argument('--save-model', type=str, default='', metavar='str',
                    help='save model file name (default: \'\')')


args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

DATA_FOLDER = args.data_folder
PRED_INPUT = args.pred_input
PRED_OUTPUT = args.pred_output
BATCH_OUT_FOLDER = args.batch_out_folder
CHANNELS = args.channels
SAVE_MODEL_NAME = args.save_model

# %% Loading in the Dataset
dset_train = BraTSDatasetUnet(DATA_FOLDER, train=True,
                              keywords=[args.modality],
                              im_size=[args.size, args.size], transform=tr.ToTensor())

train_loader = DataLoader(dset_train,
                          batch_size=args.batch_size,
                          shuffle=True, num_workers=1)

dset_test = BraTSDatasetUnet(DATA_FOLDER, train=False,
                             keywords=[args.modality],
                             im_size=[args.size, args.size], transform=tr.ToTensor())

test_loader = DataLoader(dset_test,
                         batch_size=args.test_batch_size,
                         shuffle=False, num_workers=1)

print("Data folder: ", DATA_FOLDER)
print("Load : ", args.load)
print("Training Data : ", len(train_loader.dataset))
print("Testing Data : ", len(test_loader.dataset))
print("Optimizer : ", args.optimizer)

if args.train is not True:
    dset_pred = UnetPred(PRED_INPUT, keywords=[args.modality],
                         im_size=[args.size, args.size], transform=tr.ToTensor())

    pred_loader = DataLoader(dset_pred,
                             batch_size=args.test_batch_size,
                             shuffle=False, num_workers=1)
    print("Prediction Data : ", len(pred_loader.dataset))

# %% Loading in the model
model = UNetSmall(num_channels=CHANNELS)

if args.cuda:
    model.cuda()

if args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.99)
if args.optimizer == 'ADAM':
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(0.9, 0.999))

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Defining Loss Function
criterion = DICELossMultiClass()
# Define Training Loop


def train(epoch, scheduler, loss_list):
    scheduler.step()
    model.train()
    for batch_idx, (image, mask) in enumerate(train_loader):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()

        image, mask = Variable(image), Variable(mask)

        optimizer.zero_grad()

        output = model(image)

        loss = criterion(output, mask)
        loss_list.append(loss.data[0])

        loss.backward()
        optimizer.step()

        if args.clip:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


def test(train_accuracy=False, save_output=False):
    test_loss = 0
    correct = 0

    if train_accuracy:
        loader = train_loader
    else:
        loader = test_loader

    for batch_idx, (image, mask) in tqdm(enumerate(loader)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()

        image, mask = Variable(image, volatile=True), Variable(
            mask, volatile=True)

        output = model(image)

        test_loss += criterion(output, mask).data[0]

        output.data.round_()

        if save_output and (not train_accuracy):
            np.save('./npy-files/out-files/{}-unetsmall-batch-{}-outs.npy'.format(args.save,
                                                                                  batch_idx),
                    output.data.byte().cpu().numpy())
            np.save('./npy-files/out-files/{}-unetsmall-batch-{}-masks.npy'.format(args.save,
                                                                                    batch_idx),
                    mask.data.byte().cpu().numpy())
            np.save('./npy-files/out-files/{}-unetsmall-batch-{}-images.npy'.format(args.save,
                                                                                     batch_idx),
                    image.data.float().cpu().numpy())

        if save_output and train_accuracy:
            np.save('./npy-files/out-files/{}-unetsmall-train-batch-{}-outs.npy'.format(args.save,
                                                                                        batch_idx),
                    output.data.byte().cpu().numpy())
            np.save('./npy-files/out-files/{}-unetsmall-train-batch-{}-masks.npy'.format(args.save,
                                                                                         batch_idx),
                    mask.data.byte().cpu().numpy())
            np.save('./npy-files/out-files/{}-unetsmall-train-batch-{}-images.npy'.format(args.save,
                                                                                          batch_idx),
                    image.data.float().cpu().numpy())

    # Average Dice Coefficient
    test_loss /= len(loader)
    if train_accuracy:
        print('\nTraining Set: Average DICE Coefficient: {:.4f})\n'.format(
            test_loss))
    else:
        print('\nTest Set: Average DICE Coefficient: {:.4f})\n'.format(
            test_loss))

def save_test():
    test_loss = 0

    loader = test_loader

    for batch_idx, (image, mask) in tqdm(enumerate(loader)):
        if args.cuda:
            image, mask = image.cuda(), mask.cuda()

        image, mask = Variable(image, volatile=True), Variable(
            mask, volatile=True)

        output = model(image)

        test_loss += criterion(output, mask).data[0]

        output.data.round_()

        np.save(os.path.join(BATCH_OUT_FOLDER, '{}-unetsmall-batch-{}-images.npy'.format(args.save,batch_idx)),
                image.data.float().cpu().numpy())
        np.save(os.path.join(BATCH_OUT_FOLDER, '{}-unetsmall-batch-{}-masks.npy'.format(args.save, batch_idx)),
                mask.data.byte().cpu().numpy())
        np.save(os.path.join(BATCH_OUT_FOLDER, '{}-unetsmall-batch-{}-outs.npy'.format(args.save, batch_idx)),
                output.data.byte().cpu().numpy())

    # Average Dice Coefficient
    test_loss /= len(loader)
    print('\nTest Set: Average DICE Coefficient: {:.4f})\n'.format(test_loss))

    file_names = dset_pred.get_file()
    save_dir = PRED_OUTPUT
    base_name = 'OutMasks-unetsmall'
    out_folder = BATCH_OUT_FOLDER

    save_prediction(file_names, save_dir, base_name, out_folder, True)

def predict():
    loader = pred_loader

    for batch_idx, image in tqdm(enumerate(loader)):

        if args.cuda:
            image = image.cuda()

        image= Variable(image, volatile=True)

        output = model(image)

        output.data.round_()

        np.save(os.path.join(BATCH_OUT_FOLDER, '{}-unetsmall-batch-{}-outs.npy'.format(args.save, batch_idx)),
                output.data.byte().cpu().numpy())
        np.save(os.path.join(BATCH_OUT_FOLDER, '{}-unetsmall-batch-{}-images.npy'.format(args.save,batch_idx)),
                image.data.float().cpu().numpy())

    file_names = dset_pred.get_file()
    save_dir = PRED_OUTPUT
    base_name = 'OutMasks-unetsmall'
    out_folder = BATCH_OUT_FOLDER

    save_prediction(file_names, save_dir, base_name, out_folder, False)


if args.train:
    loss_list = []
    for i in tqdm(range(args.epochs)):
        train(i, exp_lr_scheduler, loss_list)
        test(train_accuracy=False, save_output=False)
        # test(train_accuracy=True, save_output=False)

    plt.plot(loss_list)
    plt.title("UNetSmall bs={}, ep={}, lr={}".format(args.batch_size,
                                                     args.epochs, args.lr))
    plt.xlabel("Number of iterations")
    plt.ylabel("Average DICE loss per batch")
    plt.savefig("./plots/{}-UNetSmall_Loss_bs={}_ep={}_lr={}.png".format(args.save,
                                                                         args.batch_size,
                                                                         args.epochs,
                                                                         args.lr))

    np.save('./npy-files/loss-files/{}-UNetSmall_Loss_bs={}_ep={}_lr={}.npy'.format(args.save,
                                                                                    args.batch_size,
                                                                                    args.epochs,
                                                                                    args.lr),
            np.asarray(loss_list))

    torch.save(model.state_dict(), '{}unetsmall-final-{}-{}-{}'.format(SAVE_MODEL_NAME, args.batch_size,
                                                                     args.epochs,
                                                                     args.lr))
else:
    model.load_state_dict(torch.load(args.load))
    # test(save_output=True)
    # test(train_accuracy=True)
    predict()
    # save_test()
