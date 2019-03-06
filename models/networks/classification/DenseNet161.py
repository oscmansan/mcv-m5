import torchvision.models.densenet as models
from models.networks.network import Net
from torch import nn


class DenseNet161(Net):

    def __init__(self, cf, num_classes=21, pretrained=False, net_name='densenet161'):
        super(DenseNet161, self).__init__(cf)

        self.url = ''
        self.pretrained = pretrained
        self.net_name = net_name

        if pretrained:
            self.model = models.densenet161(pretrained=True)
            self.model.classifier = nn.Linear(1024, num_classes)
        else:
            self.model = models.densenet161(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model.forward(x)
