import torch
from torch import nn
import math

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


class Attention(nn.Module):
    def __init__(self,num_heads,d_model,dropout):
        super().init()
        self.num_heads=num_heads
        self.model_dim=d_model
        self.head_dim=d_model//num_heads

        self.q_proj=nn.Linear(d_model,d_model)
        self.k_proj=nn.Linear(d_model,d_model)
        self.v_proj=nn.Linear(d_model,d_model)
        self.out_proj=nn.Linear(d_model,d_model)

        self.dropput=nn.Dropout(dropout)

    def forward(self,query,key,value,padding_mask=None):
        batch_size,query_length,d_model=query.shape

        def split_heads(x):
            return x.reshape(
                batch_size,
                -1,
                self.num_heads,
                self.head_dim
            ).transpose(1,2)

        q=split_heads(self.q_proj(query))
        k=split_heads(self.k_proj(key))
        v=split_heads(self.v_proj(value))

        scores=q@k.transpose(-2,-1)/math.sqrt(self.head_dim)
        weights=self.dropout(scores.softmax(dim=-1))

        output=(weights@v).transpose(1,2).reshape(
            batch_size,
            query_length,
            d_model,
        )
        return self.out_proj(output)


