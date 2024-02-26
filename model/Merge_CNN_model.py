import os
import torch
import torch.nn as nn
from torchvision import transforms, utils, models
import torch.nn.functional as F


class CNNMerge(nn.Module):

    def __init__(self):
        super(CNNMerge, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(64)
        self.batchnorm6 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        # x = [b_s, 2, 288, 384]
        
        x = self.conv1(x)
        x = self.relu(self.batchnorm1(x))
        # x = [b_s, 32, 144, 192]
        
        x = self.conv2(x)
        x = self.relu(self.batchnorm2(x))
        # x = [b_s, 64, 144, 192]
        
        x = self.conv3(x)
        x = self.relu(self.batchnorm3(x))
        # x = [b_s, 128, 72, 96]
        
        x = self.conv4(x)
        x = self.relu(self.batchnorm4(x))
        # x = [b_s, 128, 72, 96]
        
        x = self.upsample(x)
        x = self.conv5(x)
        x = self.relu(self.batchnorm5(x))
        # x = [b_s, 64, 144, 192]
        
        x = self.upsample(x)
        x = self.conv6(x)
        x = self.relu(self.batchnorm6(x))
        # x = [b_s, 32, 288, 384]
        
        x = self.conv7(x)
        x = self.sigmoid(x)
        # x = [b_s, 1, 288, 384]
        
        return x
