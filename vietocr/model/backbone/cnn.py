import torch
from torch import nn
import math
from typing import Literal
import vietocr.model.backbone.vgg as vgg
import vietocr.model.backbone.convNext as convnet
from vietocr.model.backbone.ViT import build_sam_vit_b as vision


class CNN(nn.Module):
    def __init__(self, backbone: Literal["vgg19_bn", "convNext", "vision"], **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = "/kaggle/input/checkpoint/pytorch/default/1/pytorch_model.bin"
        if backbone == "vgg19_bn":
            self.model = vgg.vgg19_bn(**kwargs)
        elif backbone == "convNext":
            self.model = convnet.convnextv2_base(**kwargs)
        elif backbone == "vision":
            self.model = vision(**kwargs, checkpoint=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = backbone
        self.total_layers = len(list(self.get_backbone_parameters()))
        self.frozen_layers = self.total_layers - 1
        self.total_epochs = 0
        self.current_epoch = 0
        self.unfreeze_schedule = self._calculate_unfreeze_schedule()

        self.unfreeze()

    def get_backbone_parameters(self):
        if hasattr(self.model, "features"):
            return self.model.features.parameters()
        return self.model.parameters()

    def _calculate_unfreeze_schedule(self):
        speed = 3.0
        return [
            int(
                self.total_layers
                * (1 - math.exp(-speed * (i / self.total_epochs) ** 2))
            )
            for i in range(self.total_epochs)
        ]

    def forward(self, x):
        model = self.model.to(self.device)
        return model(x)

    def freeze(self):
        for i, param in enumerate(self.get_backbone_parameters()):
            param.requires_grad = i >= self.frozen_layers
        print(
            f"Partially frozen: {self.frozen_layers} layers frozen, {self.total_layers - self.frozen_layers} layers trainable."
        )

    def unfreeze(self):
        for param in self.get_backbone_parameters():
            param.requires_grad = True
        self.frozen_layers = 0
        print("All layers unfrozen.")

    def update_freeze_state(self):
        if self.current_epoch < self.total_epochs:
            target_frozen = (
                self.total_layers - self.unfreeze_schedule[self.current_epoch]
            )
            if target_frozen < self.frozen_layers:
                self.frozen_layers = target_frozen
                self.freeze()
                print(
                    f"Epoch {self.current_epoch}: Unfrozen to {self.total_layers - target_frozen} layers. {target_frozen} layers still frozen."
                )
        elif self.frozen_layers > 0:
            print("unfreeze")
            self.unfreeze()
        self.current_epoch += 1
