import torchvision.models.inception as models
from models.networks.network import Net
from torch import nn


class Inception(Net):

    def __init__(self, cf, num_classes=21, pretrained=False, net_name='inception'):
        super(Inception, self).__init__(cf)

        self.url = None
        self.pretrained = False
        self.net_name = net_name

        if pretrained:
            self.model = models.inception_v3(pretrained=True)
            self.model.AuxLogits.fc = nn.Linear(768, num_classes)
            self.model.fc = nn.Linear(2048, num_classes)
        else:
            self.model = models.inception_v3(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model.forward(x)
