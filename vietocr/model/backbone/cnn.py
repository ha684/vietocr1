import torch
from torch import nn
import vietocr.model.backbone.vgg as vgg
from vietocr.model.backbone.resnet import Resnet50
import vietocr.model.backbone.convNext as convnet
import math

class CNN(nn.Module):
    def __init__(self, backbone, total_epochs, **kwargs):
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
        
        self.total_layers = len(list(self.model.parameters()))
        self.frozen_layers = self.total_layers
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        self.unfreeze_schedule = self._calculate_unfreeze_schedule()

    def _calculate_unfreeze_schedule(self):
        schedule = []
        for i in range(self.total_epochs):
            unfrozen = int(self.total_layers * (1 - math.exp(-5 * i / self.total_epochs)))
            schedule.append(self.total_layers - unfrozen)
        return schedule

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.frozen_layers = self.total_layers
        print(f"All {self.frozen_layers} layers frozen.")

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
        self.frozen_layers = 0
        print("All layers unfrozen.")

    def update_freeze_state(self):
        if self.current_epoch < self.total_epochs:
            target_frozen = self.unfreeze_schedule[self.current_epoch]
            if target_frozen < self.frozen_layers:
                for i, param in enumerate(self.model.parameters()):
                    param.requires_grad = i >= target_frozen
                print(f"Epoch {self.current_epoch}: Unfrozen to {self.total_layers - target_frozen} layers. {target_frozen} layers still frozen.")
            self.frozen_layers = target_frozen
        else:
            if self.frozen_layers > 0:
                self.unfreeze()
        self.current_epoch += 1

    def get_optimizer(self, base_lr=1e-3, lr_multiplier=2.0):
        params = list(self.model.named_parameters())
        layer_groups = 4  # Divide the model into 4 groups
        group_size = len(params) // layer_groups

        optimizer_params = []
        for i in range(layer_groups):
            group_lr = base_lr * (lr_multiplier ** i)
            group_params = params[i*group_size:(i+1)*group_size]
            optimizer_params.append({
                'params': [p for n, p in group_params if p.requires_grad],
                'lr': group_lr
            })

        return torch.optim.AdamW(optimizer_params)
