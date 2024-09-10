import torch
import torch.nn as nn

from x_transformers import Encoder
from einops import rearrange, repeat


class ViTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        max_width,
        max_height,
        patch_size,
        attn_layers,
        channels=1,
        num_classes=None,
        dropout=0.,
        emb_dropout=0.
    ):
        super().__init__()
        assert isinstance(attn_layers, Encoder), 'attention layers must be an Encoder'
        assert max_width % patch_size == 0 and max_height % patch_size == 0, 'image dimensions must be divisible by the patch size'
        dim = attn_layers.dim
        num_patches = (max_width // patch_size)*(max_height // patch_size)
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.max_width = max_width
        self.max_height = max_height

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        #self.mlp_head = FeedForward(dim, dim_out = num_classes, dropout = dropout) if exists(num_classes) else None

    def forward(self, img, **kwargs):
        p = self.patch_size
        img = img.reshape(128,128)
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        h, w = torch.tensor(img.shape[2:])//p
        pos_emb_ind = repeat(torch.arange(h)*(self.max_width//p-w), 'h -> (h w)', w=w)+torch.arange(h*w)
        pos_emb_ind = torch.cat((torch.zeros(1), pos_emb_ind+1), dim=0).long()
        x += self.pos_embedding[:, pos_emb_ind]
        x = self.dropout(x)

        x = self.attn_layers(x, **kwargs)
        x = self.norm(x)
        x = x.transpose(-1, -2)
        x = x.flatten(2)
        x = x.permute(-1, 0, 1)
        return x


def get_encoder(channels,max_height,max_width,patch_size,dim,encoder_depth,heads):
    return ViTransformerWrapper(
        max_width=max_width,
        max_height=max_height,
        channels=channels,
        patch_size=patch_size,
        emb_dropout=0,
        attn_layers=Encoder(
            dim=dim,
            depth=encoder_depth,
            heads=heads,
        )
    )