import math

from easydict import EasyDict
from torch import nn
import torchvision.models.vgg as models

from models.networks.network import Net


class VGG16Pytorch(Net):

    def __init__(self, cf: EasyDict, num_classes: int = 21, pretrained: bool = False, net_name: str = 'vgg16'):
        super().__init__(cf)

        self.url = 'http://datasets.cvc.uab.es/models/pytorch/basic_vgg16.pth'
        self.pretrained = pretrained
        self.net_name = net_name

        self.model = models.vgg16(pretrained=False, num_classes=num_classes)

        '''if pretrained:
            self.load_basic_weights(net_name)
        else:
            self._initialize_weights()'''

    def forward(self, x):
        return self.model.forward(x)
