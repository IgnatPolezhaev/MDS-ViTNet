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
    

class CNNMerge_old(nn.Module):

    def __init__(self):
        super(CNNMerge_old, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv8 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv9 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv10 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv11 = nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        

        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.batchnorm6 = nn.BatchNorm2d(64)
        self.batchnorm7 = nn.BatchNorm2d(64)
        self.batchnorm8 = nn.BatchNorm2d(32)
        self.batchnorm9 = nn.BatchNorm2d(32)
        self.batchnorm10 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        
        x1 = self.relu(self.batchnorm1(x1))
        x2 = self.relu(self.batchnorm2(x2))
        
        x = torch.cat((x1, x2), 1)

        x = self.relu(self.dropout(self.batchnorm3(self.conv3(x))))
        x = self.relu(self.dropout(self.batchnorm4(self.conv4(x))))
        x = self.relu(self.dropout(self.batchnorm5(self.conv5(x))))
        x = self.relu(self.dropout(self.batchnorm6(self.conv6(x))))
        x = self.relu(self.dropout(self.batchnorm7(self.conv7(x))))
        x = self.relu(self.dropout(self.batchnorm8(self.conv8(x))))
        x = self.relu(self.dropout(self.batchnorm9(self.conv9(x))))
        x = self.relu(self.dropout(self.batchnorm10(self.conv10(x))))

        x = self.sigmoid(self.conv11(x))
        
        return x
    

class CNNMerge_6(nn.Module):

    def __init__(self):
        super(CNNMerge_6, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.batchnorm5 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = x1 * x2
        
        x = self.relu(self.batchnorm1(self.conv1(x)))
        # x = [b_s, 32, 288, 384]
        
        x = self.relu(self.batchnorm2(self.conv2(x)))
        # x = [b_s, 64, 144, 192]
        
        x = self.relu(self.batchnorm3(self.conv3(x)))
        # x = [b_s, 128, 144, 192]
        
        x = self.relu(self.batchnorm4(self.conv4(x)))
        # x = [b_s, 64, 144, 192]
        
        x = self.upsample(x)
        # x = [b_s, 64, 288, 384]
        
        x = self.relu(self.batchnorm5(self.conv5(x)))
        # x = [b_s, 32, 288, 384]
        
        x = self.sigmoid(self.conv6(x))
        # x = [b_s, 1, 288, 384]
        
        return x

    
class CNNMerge_7(nn.Module):

    def __init__(self):
        super(CNNMerge_7, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(34, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.batchnorm4 = nn.BatchNorm2d(96)
        self.batchnorm5 = nn.BatchNorm2d(32)

        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        
        x1_conv = self.relu(self.dropout(self.batchnorm1(self.conv1(x1))))
        x2_conv = self.relu(self.dropout(self.batchnorm2(self.conv2(x2))))
        # x1_conv, x2_conv = [b_s, 32, 144, 192]
        
        x12 = x1 * x2
        x12_conv = self.relu(self.dropout(self.batchnorm3(self.conv3(x12))))
        # x12_conv = [b_s, 32, 144, 192]
        
        x_add = torch.cat((x1_conv, x2_conv, x12_conv), 1)
        # x_add = [b_s, 96, 144, 192]
        
        x = self.relu(self.batchnorm4(self.conv4(x_add)))
        # x = [b_s, 96, 144, 192]
        
        x = self.upsample(x)
        x = self.relu(self.batchnorm5(self.conv5(x)))
        # x = [b_s, 32, 288, 384]
        
        x_final = torch.cat((x1, x2, x), 1)
        # x_final = [b_s, 34, 288, 384]
        
        x = self.sigmoid(self.conv6(x_final))
        # x = [b_s, 32, 288, 384]
        
        return x
    
    
class CNNMerge_8(nn.Module):

    def __init__(self):
        super(CNNMerge_8, self).__init__()

        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(384, 384)
        
        self.batchnorm = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.2)                                            
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        
        x = torch.cat((x1, x2), 3)
        # x = [b_s, 1, 288, 768]
        
        x = self.relu(self.linear1(x))
        x = self.sigmoid(self.linear2(x))

        return x