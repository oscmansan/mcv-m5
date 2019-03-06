import torchvision.models.vgg as models
from easydict import EasyDict
from torch import nn

from models.networks.network import Net


class VGG16Torch(Net):

    def __init__(self, cf: EasyDict, num_classes: int = 21, pretrained: bool = False, net_name: str = 'vgg16torch'):
        super().__init__(cf)

        self.url = 'http://datasets.cvc.uab.es/models/pytorch/basic_vgg16.pth'
        self.pretrained = pretrained
        self.net_name = net_name

        self.model = models.vgg16(pretrained=False, num_classes=num_classes)

        if pretrained:
            self.model = models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, num_classes)
        else:
            self.model = models.vgg16(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model.forward(x)
