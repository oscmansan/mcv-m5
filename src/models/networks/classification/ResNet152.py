import torchvision.models.resnet as models
from models.networks.network import Net
from torch import nn


class ResNet152(Net):

    def __init__(self, cf, num_classes=21, pretrained=False, net_name='resnet152'):
        super(ResNet152, self).__init__(cf)

        self.pretrained = pretrained
        self.net_name = net_name

        if pretrained:
            self.model = models.resnet152(pretrained=True)
            self.model.fc = nn.Linear(512, num_classes)
        else:
            self.model = models.resnet152(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model.forward(x)

    def load_basic_weights(self):
        pass
