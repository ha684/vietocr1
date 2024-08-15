import torch
import torch.nn as nn
from timm import create_model

class ConvNeXtV2(nn.Module):
    def __init__(self, model_name='convnextv2_base', pretrained=True):
        super().__init__()
        self.model = create_model(model_name, pretrained=pretrained)
        self.model.head = nn.Identity()

    def forward(self, x):
        x = self.model.forward_features(x) 
        return x
def convnextv2():
    return ConvNeXtV2()
if __name__ == "__main__":
    model = ConvNeXtV2(model_name='convnextv2_base', pretrained=True)
    input_tensor = torch.randn(1, 3, 224, 224) 
    features = model(input_tensor)
    print(features.shape)
