import torch
from torch import nn

import vietocr.model.backbone.vgg as vgg
from vietocr.model.backbone.resnet import Resnet50
from vietocr.model.backbone.convNext import convnextv2_base
# from timm.models.vision_transformer_hybrid import vit_small_r26_s32_224_in21k
# from vietocr.model.backbone.vision import mamba_vision_B
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
            self.model = convnextv2_base(**kwargs)
        # elif backbone == 'vision_transformer':
        #     self.model = vit_small_r26_s32_224_in21k(pretrained=True,**kwargs)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
