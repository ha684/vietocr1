import torch
from torch import nn
# from vit_pytorch import ViT
import vietocr.model.backbone.vgg as vgg
from vietocr.model.backbone.resnet import Resnet50
import vietocr.model.backbone.convNext as convnet
# from vietocr.model.backbone.vision import get_encoder
# from timm.models.vision_transformer_hybrid import vit_small_r26_s32_224_in21k
# from vietocr.model.backbone.vision import mamba_vision_B
class CNN(nn.Module):
    def __init__(self, backbone, **kwargs):
        super(CNN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if backbone == 'vgg11_bn':
            self.model = vgg.vgg11_bn(**kwargs)
        elif backbone == 'vgg19_bn':
            self.model = vgg.vgg19_bn(**kwargs)
        elif backbone == 'resnet50':
            self.model = Resnet50(**kwargs)
        elif backbone == 'convNext':
            self.model = convnet.convnextv2_base(**kwargs)
        # elif backbone == 'vision_transformer':
        #     self.model = ViT(**kwargs)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for name, param in self.model.features.named_parameters():
            if name != 'last_conv_1x1':
                param.requires_grad = False

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True
