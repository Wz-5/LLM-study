import torch
from torch import nn

class selfAttention(nn.Module):
    def __init__(self,model_dim,num_heads):
        super().__init_()
        self.qkv_proj=nn.Linear(model_dim,model_dim*3)
        self.num_heads=num_heads

    def forward(self,x):
        qkv=self.qkv_proj(x)
        q,k,v=qkv.chunk(3,dim=-1)
        scores=q.matmul(k.tanspose(-2,-1))/q.size(-1)**0.5
        attn=scores.softmax(dim=-1)
        out=attn.matmul(v)
        return out
    
