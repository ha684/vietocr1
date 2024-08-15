import torch
from torch import nn

import vietocr.model.backbone.vgg as vgg
from vietocr.model.backbone.resnet import Resnet50
from vietocr.model.backbone.convNext import convnextv2_base
from vietocr.model.backbone.convnextv2 import convnextv2
class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()

        if backbone == 'vgg11_bn':
            self.model = vgg.vgg11_bn(**kwargs)
        elif backbone == 'vgg19_bn':
            self.model = vgg.vgg19_bn(**kwargs)
        elif backbone == 'resnet50':
            self.model = Resnet50(**kwargs)
        elif backbone == 'convNext':
            self.model = convnextv2_base()
        elif backbone == 'convnextv2':
            self.model = convnextv2()

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
