import torch
from torch import nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim,dropout=0.):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU()
        )
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, dim)
        )

    def forward(self,x):
        x = self.net1(x)
        mask1 = torch.ones_like(x) * 0.3
        x = x + torch.bernoulli(mask1) * -1e12
        x = self.net2(x)
        mask2 = torch.ones_like(x) * 0.1
        x = x + torch.bernoulli(mask2) * -1e12
        return x

class Flat1D(nn.Module):
    def __init__(self,in_channel = 512):
        super().__init__()
        self.proj = nn.Conv2d(in_channel,in_channel,kernel_size= 1)

    def forward(self,x):
        B,C,H,W = x.shape
        x = self.proj(x).flatten(2).transpose(1,2)
        return x

class Attention(nn.Module):
    def __init__(self,dim = 512,head = 8,head_dim = 64,dropout = 0.):
        super().__init__()
        self.heads = head
        self.dim = dim
        self.head_dim = head_dim
        self.scale = head_dim ** (-0.5)
        self.dropout = nn.Dropout(dropout)
        self.qkv = nn.Linear(dim,dim * 3)
        self.proj = nn.Linear(dim,dim)

    def forward(self,x):
        B,N,C = x.shape
        # reshape:[B,N,3 * C] -> [B,N,3,head,C // head]
        # permute:[B,N,3,head,C // head] -> [3,B,head,N,C // head]
        qkv = self.qkv(x).reshape(B,N,3,self.heads,C // self.heads).permute(2,0,3,1,4)
        # q,k,v -> [B,head,N,C // head]
        q,k,v = qkv[0],qkv[1],qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        m_r = torch.ones_like(attn) * 0.5
        attn = attn + torch.bernoulli(m_r) * -1e12
        attn = attn.softmax(dim = -1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1,2).reshape(B,N,C)
        x = self.proj(x)
        return x

class MHA(nn.Module):
    def __init__(self,in_channel = 512,head = 8,head_dim = 64,dropout = 0.,out_channel = 2560):
        super().__init__()
        self.embed = Flat1D(in_channel)
        self.attn = Attention(in_channel,head,head_dim,dropout)
        self.net = FeedForward(in_channel,in_channel,dropout)
        self.norm = nn.LayerNorm(in_channel)
        self.fc = nn.Linear(in_channel,out_channel)
    def forward(self,x):
        x = self.embed(x)
        x = self.norm(x)
        x = self.attn(x)
        x = self.net(x)
        x = self.norm(x)
        x = x.mean(dim = 1)
        x = self.fc(x)
        return x



