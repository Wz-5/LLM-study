import torch
from torch import nn

def window_partition(x, window_size):
    
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    C=windows.shape[-1]
    x = windows.reshape(-1, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, H, W, C)
    return x

def shift_mask(H,W,m,shift,device,dtype):

    coords=torch.stack(torch.meshgrid(torch.arange(H),torch.arange(W), indexing="ij"), dim=-1).to(device=device,dtype=dtype)
    coords=torch.roll(coords, shifts=(-shift,-shift), dims=(0,1))
    coords=window_partition(coords.unsqueeze(0), m)

    distance=coords[:, :, None, :] - coords[:, None,  :, :]
    distance=distance.abs()
    blocked=distance(distance>=m).any(dim=-1)

    return torch.zeros(blocked.shape, device=device, dtype=dtype).masked_fill(blocked, -100)
class WindowAttenton(nn.Moudle):
    
    def __init__(self,dim,num_heads,window_size=7,attn_drop=0.,proj_drop=0.):
        super().__init__()
        self.dim=dim
        self.num_heads=num_heads
        self.window_size=window_size
        self.scale=(dim//num_heads)**-0.5

        self.qkv=nn.Linear(dim,dim*3,bias=True)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)

        self.bias_table=nn.Parameter(torch.zeros((2*window_size-1)*(2*window_size-1),num_heads))
        coords=torch.stack(torch.meshgrid(torch.arange(window_size),torch.arange(window_size), indexing="ij"), dim=-1).flatten(1)
        offsets=coords[:,None,:]-coords[None,:,:]
        index=(offsets[0]+window_size-1)*(2*window_size-1)+offsets[1]+window_size-1
        self.register_buffer("index",index)
        nn.init.trunc_normal_(self.bias_table,std=.02)

    def forward(self,x,mask=None):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]

        scores=(q @ k.transpose(-2,-1))*self.scale
        bias=self.bias_table[self.index.view(-1)].view(self.window_size*self.window_size,self.window_size*self.window_size,-1)
        bias=bias.reshape(N,N,self.heads).permute(2,0,1)
        scores=scores+bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            scores = scores.reshape(
                -1, nW, self.heads, N, N
            )
            scores = scores + mask[None, :, None, :, :]
            scores = scores.reshape(B, self.heads, N, N)

        weights = self.attn_drop(scores.softmax(dim=-1))
        x = (weights @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


        
        
