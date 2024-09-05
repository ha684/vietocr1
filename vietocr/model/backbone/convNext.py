import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from vietocr.model.backbone.convnext_v2.models.utils import LayerNorm, GRN
import timm

class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        print(x.shape,input.shape)
        x = input + self.drop_path(x)
        return x
    
class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.,hidden=512,
                 ks=None,ss=None,
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )

        self.downsample_layers.append(stem)
        pool_idx = 0  
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.AvgPool2d(kernel_size=ks[pool_idx], stride=ss[pool_idx], padding=0),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=1, stride=1),
                    )
            self.downsample_layers.append(downsample_layer)
            pool_idx += 1

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Conv2d(dims[-1], hidden,1)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x) 
        x = x.transpose(-1, -2)
        x = x.flatten(2)
        x = x.permute(-1, 0, 1)
        return x
    
def convnextv2_base(pretrained=True,**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    if pretrained:
        temp_model = timm.create_model('convnextv2_pico', pretrained=True)
        state_dict = temp_model.state_dict()
        model.load_state_dict(state_dict, strict=False)
    model.to('cuda')
    return model

if __name__ == "__main__":
    model = convnextv2_base(pretrained=True)
    input_tensor = torch.randn(1, 3, 224, 224).to('cuda')
    features = model(input_tensor)
    print(features.shape)
