"""
Reference:
https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/models/psp.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks.network import Net
from models.networks.segmentation.resnet import resnet50


class PSP_head(nn.Module):

    def __init__(self, in_channels):

        super(PSP_head, self).__init__()

        out_channels = int(in_channels / 4)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

        self.fusion_bottleneck = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(out_channels),
                                               nn.ReLU(True),
                                               nn.Dropout2d(0.1, False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        fcn_features_spatial_dim = x.size()[2:]

        pooled_1 = F.adaptive_avg_pool2d(x, 1)
        pooled_1 = self.conv1(pooled_1)
        pooled_1 = F.interpolate(pooled_1, size=fcn_features_spatial_dim, mode='bilinear', align_corners=True)

        pooled_2 = F.adaptive_avg_pool2d(x, 2)
        pooled_2 = self.conv2(pooled_2)
        pooled_2 = F.interpolate(pooled_2, size=fcn_features_spatial_dim, mode='bilinear', align_corners=True)

        pooled_3 = F.adaptive_avg_pool2d(x, 3)
        pooled_3 = self.conv3(pooled_3)
        pooled_3 = F.interpolate(pooled_3, size=fcn_features_spatial_dim, mode='bilinear', align_corners=True)

        pooled_4 = F.adaptive_avg_pool2d(x, 6)
        pooled_4 = self.conv4(pooled_4)
        pooled_4 = F.interpolate(pooled_4, size=fcn_features_spatial_dim, mode='bilinear', align_corners=True)

        x = torch.cat((x, pooled_1, pooled_2, pooled_3, pooled_4), dim=1)

        x = self.fusion_bottleneck(x)

        return x


class PSP_Resnet50_8s(Net):

    def __init__(self, cf, num_classes=1000, pretrained=False, net_name='PSPNet'):
        super(PSP_Resnet50_8s, self).__init__(cf)
        self.pretrained = pretrained
        self.net_name = net_name

        # Load the pretrained weights, remove avg pool layer and get the output stride of 16
        resnet50_8s = resnet50(fully_conv=True,
                               pretrained=pretrained,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        self.psp_head = PSP_head(resnet50_8s.inplanes)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes // 4, num_classes, 1)

        self.resnet50_8s = resnet50_8s

    def forward(self, x):
        input_spatial_dim = x.size()[2:]

        x = self.resnet50_8s.conv1(x)
        x = self.resnet50_8s.bn1(x)
        x = self.resnet50_8s.relu(x)
        x = self.resnet50_8s.maxpool(x)

        x = self.resnet50_8s.layer1(x)
        x = self.resnet50_8s.layer2(x)
        x = self.resnet50_8s.layer3(x)
        x = self.resnet50_8s.layer4(x)

        x = self.psp_head(x)

        x = self.resnet50_8s.fc(x)

        x = F.interpolate(input=x, size=input_spatial_dim, mode='bilinear', align_corners=True)

        return x

    def initialize_weights(self):
        pass

    def load_basic_weights(self):
        pass
