import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from PIL import Image
import torch.nn as nn
from torchvision import transforms, utils, models
import torch.nn.functional as F
#import utils.densenet as densenet
from torchvision.models import swin_t

from utils.TransformerEncoder import Encoder


cfg6 = {
"hidden_size" : 768,
"mlp_dim" : 768*4,
"num_heads" : 12,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}

cfg5 = {
"hidden_size" : 768,
"mlp_dim" : 768*4,
"num_heads" : 12,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}

cfg4 = {
"hidden_size" : 768,
"mlp_dim" : 768*4,
"num_heads" : 12,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}

cfg3 = {
"hidden_size" : 768,
"mlp_dim" : 768*4,
"num_heads" : 12,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}

cfg2 = {
"hidden_size" : 512,
"mlp_dim" : 512*4,
"num_heads" : 8,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}

cfg1 = {
"hidden_size" : 512,
"mlp_dim" : 512*4,
"num_heads" : 8,
"num_layers" : 2,
"attention_dropout_rate" : 0,
"dropout_rate" : 0.0,
}


class TranSalNet(nn.Module):

    def __init__(self):
        super(TranSalNet, self).__init__()
        self.encoder = _Encoder()
        self.decoder_1 = _Decoder_1()
        self.decoder_2 = _Decoder_2()

    def forward(self, x):
        x1, x2 = self.encoder(x)
        x1 = self.decoder_1(x1)
        x2 = self.decoder_2(x2)
        return x1, x2


class _Encoder(nn.Module):

    def __init__(self):
        super(_Encoder, self).__init__()
        base_model = swin_t(pretrained=True)
        base_layers = list(base_model.children())[0][:-1]
        self.encoder = nn.ModuleList(base_layers).eval()

    def forward(self, x):
        outputs_1, outputs_2 = [], []
        for ii, layer in enumerate(self.encoder):
            x = layer(x)
            if ii in {2, 4, 6}:
                outputs_1.append(x)
            elif ii in {1, 3, 5}:
                outputs_2.append(x)
        return outputs_1, outputs_2
    

class _Decoder_1(nn.Module):

    def __init__(self):
        super(_Decoder_1, self).__init__()
        self.conv1 = nn.Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm1 = nn.BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm6 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.TransEncoder2 = TransEncoder(in_channels=192, spatial_size=36*48, cfg=cfg2)
        self.TransEncoder4 = TransEncoder(in_channels=384, spatial_size=18*24, cfg=cfg4)
        self.TransEncoder6 = TransEncoder(in_channels=768, spatial_size=9*12, cfg=cfg6)

        self.add = torch.add
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x2, x4, x6 = x
        # x2 = [b_s, 36, 48, 192], x4 = [b_s, 18, 24, 384], x6 = [b_s, 9, 12, 768]
        x2 = x2.permute(0,3,1,2)
        x4 = x4.permute(0,3,1,2)
        x6 = x6.permute(0,3,1,2)
        # x2 = [b_s, 192, 36, 48], x4 = [b_s, 384, 18, 24], x6 = [b_s, 768, 9, 12]
        
        x6 = self.TransEncoder6(x6)
        # x6 = [b_s, 768, 9, 12]
        x6 = self.conv1(x6)
        x6 = self.batchnorm1(x6)
        x6 = self.relu(x6)
        x6 = self.upsample(x6)
        # x6 = [b_s, 768, 18, 24]

        x4_a = self.TransEncoder4(x4)
        # x4_a = [b_s, 768, 18, 24]
        x4 = x6 * x4_a
        x4 = self.relu(x4)
        x4 = self.conv2(x4)
        x4 = self.batchnorm2(x4)
        x4 = self.relu(x4)
        x4 = self.upsample(x4)
        # x4 = [b_s, 512, 36, 48]

        x2_a = self.TransEncoder2(x2)
        # x2_a = [b_s, 512, 36, 48]
        x2 = x4 * x2_a
        x2 = self.relu(x2)
        x2 = self.conv3(x2)
        x2 = self.batchnorm3(x2)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)
        # x2 = [b_s, 256, 72, 96]

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.sigmoid(x)

        return x
    

class _Decoder_2(nn.Module):

    def __init__(self):
        super(_Decoder_2, self).__init__()
        self.conv1 = nn.Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.batchnorm1 = nn.BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm5 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batchnorm6 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.TransEncoder1 = TransEncoder(in_channels=96, spatial_size=72*96, cfg=cfg1)
        self.TransEncoder3 = TransEncoder(in_channels=192, spatial_size=36*48, cfg=cfg3)
        self.TransEncoder5 = TransEncoder(in_channels=384, spatial_size=18*24, cfg=cfg5)

        self.add = torch.add
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x3, x5 = x
        # x1 = [b_s, 72, 96, 96], x3 = [b_s, 36, 48, 192], x5 = [b_s, 18, 24, 384]
        x1 = x1.permute(0,3,1,2)
        x3 = x3.permute(0,3,1,2)
        x5 = x5.permute(0,3,1,2)
        # x1 = [b_s, 96, 72, 96], x3 = [b_s, 192, 36, 48], x5 = [b_s, 384, 18, 24]

        x5 = self.TransEncoder5(x5)
        # x5 = [b_s, 768, 18, 24]
        x5 = self.conv1(x5)
        x5 = self.batchnorm1(x5)
        x5 = self.relu(x5)
        x5 = self.upsample(x5)
        # x5 = [b_s, 768, 18, 24]

        x3_a = self.TransEncoder3(x3)
        # x3_a = [b_s, 768, 18, 24]
        x3 = x5 * x3_a
        x3 = self.relu(x3)
        x3 = self.conv2(x3)
        x3 = self.batchnorm2(x3)
        x3 = self.relu(x3)
        x3 = self.upsample(x3)
        # x3 = [b_s, 512, 36, 48]

        x1_a = self.TransEncoder1(x1)
        # x1_a = [b_s, 512, 36, 48]
        x1 = x3 * x1_a
        x1 = self.relu(x1)
        x1 = self.conv3(x1)
        x1 = self.batchnorm3(x1)
        x1 = self.relu(x1)
        x1 = self.upsample(x1)
        # x1 = [b_s, 256, 72, 96]

        x = self.conv4(x1)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.sigmoid(x)

        return x


class TransEncoder(nn.Module):

    def __init__(self, in_channels, spatial_size, cfg):
        super(TransEncoder, self).__init__()

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                          out_channels=cfg['hidden_size'],
                                          kernel_size=1,
                                          stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, spatial_size, cfg['hidden_size']))

        self.transformer_encoder = Encoder(cfg)

    def forward(self, x):
        a, b = x.shape[2], x.shape[3]
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        x = self.transformer_encoder(embeddings)
        B, n_patch, hidden = x.shape
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, a, b)

        return x
