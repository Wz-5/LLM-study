import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                 in_chans=3, dim=768):
        super().__init__()
        

        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        

        x = self.proj(x)                 
        return x.flatten(2).transpose(1, 2)  


class Attention(nn.Module):
    def __init__(self, dim=768, num_heads=12,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(
            B, T, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)       
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        
        x = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj_drop(self.proj(x))


