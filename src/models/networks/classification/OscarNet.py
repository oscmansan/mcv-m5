from torch import nn
import torch.nn.functional as F

from models.networks.network import Net


class Conv2dBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2dBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        return x


class OscarNet(Net):

    def __init__(self, cf, num_classes=45, pretrained=False, net_name='oscarnet'):
        super(OscarNet, self).__init__(cf)

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.net_name = net_name

        self.inplanes = 32
        self.init_conv = Conv2dBN(3, self.inplanes, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(32, repetitions=2)
        self.layer2 = self._make_layer(64, repetitions=2)
        self.layer3 = self._make_layer(128, repetitions=3)
        self.final_conv = Conv2dBN(128, num_classes, kernel_size=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def _make_layer(self, planes, repetitions):
        layers = []
        layers.append(Conv2dBN(self.inplanes, planes, kernel_size=3))
        self.inplanes = planes
        for _ in range(1, repetitions):
            layers.append(Conv2dBN(self.inplanes, planes, kernel_size=3))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.dropout(x, p=0.5)
        x = self.final_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), self.num_classes)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
