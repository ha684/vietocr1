import torch
from torch import nn
from timm import create_model

class ConvNeXtV2(nn.Module):
    def __init__(self, model_name, hidden=256, pretrained=True, dropout=0.5):
        super(ConvNeXtV2, self).__init__()
        
        # Load ConvNeXt V2 model
        self.model = create_model(model_name, pretrained=pretrained, features_only=True)
        
        # Extract the number of channels in the last layer of ConvNeXt V2
        last_layer_channels = self.model.feature_info[-1]['num_chs']
        
        # Custom pooling layers reflecting VGG configurations
        self.pool_layers = nn.ModuleList([
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [2, 2]
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),  # [2, 2]
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [2, 1]
            nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),  # [2, 1]
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))   # [1, 1]
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.last_conv_1x1 = nn.Conv2d(last_layer_channels, hidden, 1)
        
    def forward(self, x):
        # Extract features using ConvNeXt V2
        conv = self.model(x)[-1]
        
        # Apply custom pooling layers in sequence
        for pool in self.pool_layers:
            conv = pool(conv)
        
        conv = self.dropout(conv)
        conv = self.last_conv_1x1(conv)

        # Reshape and permute the output as required
        conv = conv.transpose(-1, -2)
        conv = conv.flatten(2)
        conv = conv.permute(-1, 0, 1)
        return conv

def convnextv2(hidden=256, pretrained=True, dropout=0.5, model_name='convnextv2_base'):
    return ConvNeXtV2(model_name, hidden, pretrained, dropout)
