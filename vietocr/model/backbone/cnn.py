import torch
from torch import nn
import math
from typing import Literal
import vietocr.model.backbone.vgg as vgg
import vietocr.model.backbone.convNext as convnet
from vietocr.model.backbone.ViT import build_sam_vit_b as vision

class CNN(nn.Module):
    def __init__(self, backbone: Literal['vgg19_bn', 'convNext', 'vision'], **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = '/kaggle/input/checkpoint/pytorch/default/1/pytorch_model.bin'
        if backbone == 'vgg19_bn':
            self.model = vgg.vgg19_bn(**kwargs)
        elif backbone == 'convNext':
            self.model = convnet.convnextv2_base(**kwargs)
        elif backbone == 'vision':
            self.model = vision(**kwargs,checkpoint=checkpoint)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.backbone = backbone
        self.total_layers = len(list(self.get_backbone_parameters()))
        self.frozen_layers = self.total_layers - 1
        self.total_epochs = 50000
        self.current_epoch = 0
        self.unfreeze_schedule = self._calculate_unfreeze_schedule()
        
        self.freeze()
    
    def get_backbone_parameters(self):
        if hasattr(self.model, 'features'):
            return self.model.features.parameters()
        return self.model.parameters()

    def _calculate_unfreeze_schedule(self):
        speed = 3.0
        return [
            int(self.total_layers * (1 - math.exp(-speed * (i / self.total_epochs) ** 2)))
            for i in range(self.total_epochs)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.model.to(self.device)
        return model(x)

    def freeze(self):
        for i, param in enumerate(self.get_backbone_parameters()):
            param.requires_grad = i >= self.frozen_layers
        print(f"Partially frozen: {self.frozen_layers} layers frozen, {self.total_layers - self.frozen_layers} layers trainable.")

    def unfreeze(self):
        for param in self.get_backbone_parameters():
            param.requires_grad = True
        self.frozen_layers = 0
        print("All layers unfrozen.")

    def update_freeze_state(self):
        if self.current_epoch < self.total_epochs:
            target_frozen = self.total_layers - self.unfreeze_schedule[self.current_epoch]
            if target_frozen < self.frozen_layers:
                self.frozen_layers = target_frozen
                self.freeze()
                print(f"Epoch {self.current_epoch}: Unfrozen to {self.total_layers - target_frozen} layers. {target_frozen} layers still frozen.")
        elif self.frozen_layers > 0:
            print('unfreeze')
            self.unfreeze()
        self.current_epoch += 1

    def get_optimizer(self, base_lr: float = 1e-4, lr_multiplier: float = 0.5):
        def belongs_to_stage(param_name, prefixes):
            return any(param_name.startswith(prefix) for prefix in prefixes)
        
        if hasattr(self.model, 'features'):
            params = list(self.model.features.named_parameters())
            if self.backbone == 'vgg19_bn':
                stage_prefixes = {
                    'stage1': ['1.'],
                    'stage2': ['2.'],
                    'stage3': ['3.'],
                    'stage4': ['4.'],
                    'stage5': ['5.'],
                    'stage6': ['6.'],
                    'stage7': ['7.']
                }
            elif self.backbone == 'convNext':
                stage_prefixes = {
                    'stage1': ['downsample_layers.0.', 'stages.0.'],
                    'stage2': ['downsample_layers.1.', 'stages.1.'],
                    'stage3': ['downsample_layers.2.', 'stages.2.'],
                    'stage4': ['downsample_layers.3.', 'stages.3.']
                }
        else:  # For ViT or other models without 'features'
            params = list(self.model.named_parameters())
            # Adjust this according to your ViT model structure
            stage_prefixes = {
                'stage1': ['encoder.blocks.0.'],
                'stage2': ['encoder.blocks.1.'],
                'stage3': ['encoder.blocks.2.'],
                # Add more stages as needed
            }

        param_groups = []
        for i, (stage_name, prefixes) in enumerate(stage_prefixes.items()):
            stage_params = [p for n, p in params if belongs_to_stage(n, prefixes)]
            if stage_params:
                lr = base_lr * (lr_multiplier ** i)
                param_groups.append({
                    'params': stage_params,
                    'lr': lr
                })
        
        other_params = [p for n, p in params if not any(belongs_to_stage(n, prefixes) for prefixes in stage_prefixes.values())]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr 
            })
        
        return torch.optim.AdamW(
            param_groups, 
            betas=(0.9, 0.98), 
            eps=1e-09,
            weight_decay=0.001
        )

    def report_layers(self):
        print("\nLayer Report:")
        total_params = 0
        trainable_params = 0
        for i, (name, param) in enumerate(self.get_backbone_parameters()):
            params = param.numel()
            status = "Trainable" if param.requires_grad else "Frozen"
            print(f"Layer {i}: {name}, Parameters: {params}, Status: {status}")
            total_params += params
            if param.requires_grad:
                trainable_params += params
        
        print(f"\nTotal Parameters: {total_params}")
        print(f"Trainable Parameters: {trainable_params}")
        print(f"Percentage Trainable: {trainable_params/total_params*100:.2f}%\n")