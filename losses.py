# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.functional as f
import numpy as np


class DICELossMultiClass(nn.Module):

    def __init__(self):
        super(DICELossMultiClass, self).__init__()

    def forward(self, output, mask):

        probs = output[:, 1, :, :]
        mask = torch.squeeze(mask, 1)

        num = probs * mask
        num = torch.sum(num, 2)
        num = torch.sum(num, 1)

        # print( num )

        den1 = probs * probs
        # print(den1.size())
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)

        # print(den1.size())

        den2 = mask * mask
        # print(den2.size())
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)

        # print(den2.size())
        eps = 0.0000001
        dice = 2 * ((num + eps) / (den1 + den2 + eps))
        # dice_eso = dice[:, 1:]
        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss


class DICELoss(nn.Module):

    def __init__(self):
        super(DICELoss, self).__init__()

    def forward(self, output, mask):

        probs = torch.squeeze(output, 1)
        mask = torch.squeeze(mask, 1)

        intersection = probs * mask
        intersection = torch.sum(intersection, 2)
        intersection = torch.sum(intersection, 1)

        # print( num )

        den1 = probs * probs
        # print(den1.size())
        den1 = torch.sum(den1, 2)
        den1 = torch.sum(den1, 1)

        # print(den1.size())

        den2 = mask * mask
        # print(den2.size())
        den2 = torch.sum(den2, 2)
        den2 = torch.sum(den2, 1)

        # print(den2.size())
        eps = 0.0000001
        dice = 2 * ((intersection + eps) / (den1 + den2 + eps))
        # dice_eso = dice[:, 1:]
        dice_eso = dice

        loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
        return loss

class DICELoss3D(nn.Module):

    def __init__(self):
        super(DICELoss3D, self).__init__()

    def forward(self, output, mask):

        batch_size, channel, x, y, z = output.size()
        total_loss = 0
        for i in range(batch_size):
            for j in range(z):
                loss = 0
                output_z = output[i:i + 1, :, :, :, j]
                label_z = mask[i, :, :, :, j]

                softmax_output_z = nn.Softmax2d()(output_z)
                logsoftmax_output_z = torch.log(softmax_output_z)

                loss = nn.NLLLoss2d()(logsoftmax_output_z, label_z)
                total_loss += loss

        return total_loss
