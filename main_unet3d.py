import torch
import torch.nn as nn
from torch.autograd import Variable

from models import Unet_3D


image_size = 64
x = Variable(torch.Tensor(1, 3, image_size, image_size, image_size))
print(x.size())

model = nn.Conv3d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True)
print(model(x).size())

unet = Unet_3D(in_dim=3, out_dim=3, num_features=4)
print(unet(x).size())
