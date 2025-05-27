import pdb

import torch
from torch import nn

from einops import rearrange,repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t,tuple) else (t,t)

class PreNorm(nn.Module):
    def __init__(self, dim,fn):
        super().__init__()
        self.norm =nn.LayerNorm(dim)
        self.fn =fn
    def forward(self,x,**kwargs):
        return self.fn(self.norm(x),**kwargs)
    

def posemb_sincos_2d(patches,temperature = 10000,dtype = torch.float32):
    _,h,w,dim,device,dtype = *patches.shape,patches.device,patches.dtype
    y,x =torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing= 'ij') #生成维度为（h,w）的两个网格
    assert (dim%4)==0, 'feature dimension must be multiple of 4 for sincos emb'
    omega =torch.arange(dim // 4,device=device) / (dim // 4-1)
    omega = 1. / (temperature ** omega)

    y =y.flatten()[:,None] * omega[None ,:]
    x =x.flatten()[:,None] * omega[None ,:]

    pe =torch.cat((x.sin(),x.cos(),y.sin(),y.cos()),dim=-1)
    return pe.type(dtype)
###  VIT1
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


 ### vit 1
class Attention(nn.Module):
    def __init__(self, dim, heads=8,dim_head =320,dropout =0.):
        super().__init__()
        inner_dim =dim_head * heads

        project_out = not (heads ==1 and dim_head ==dim)

        self.heads =heads
        self.scale =dim_head ** -0.5
        self.dropout =nn.Dropout(dropout)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv =nn.Linear(dim,inner_dim*3,bias= False)

        self.to_out =nn.Sequential(
            nn.Linear(inner_dim,dim),
            nn.Dropout(dropout)
        )if project_out else nn.Identity()
    def forward(self,x):

        qkv =self.to_qkv(x).chunk(3,dim=-1)
        q,k,v =map(lambda t: rearrange(t,'b n (h d) -> b h n d',h =self.heads),qkv)

        dots =  torch.matmul(q,k.transpose(-1,-2))*self.scale

        m_r = torch.ones_like(dots) * 0.5
        dots = dots + torch.bernoulli(m_r) * -1e12

        attn =self.attend(dots)


        out =torch.matmul(attn,v)
        out = rearrange(out,'b h n d -> b n (h d)')

        return self.to_out(out)  

class Transformer(nn.Module):
    def __init__(self, dim,depth,heads,dim_head,mlp_dim,dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [
                    PreNorm(dim,Attention(dim,heads=heads,dim_head=dim_head,dropout= dropout)),
                    PreNorm(dim,FeedForward(dim,mlp_dim,dropout=dropout))
                ]
            ))
    def forward(self,x):
        for attn ,ff in self.layers:
            x= attn(x) +x
            x =ff(x)+x
        return x
class VIT(nn.Module):
    def __init__(self, *, image_size,patch_size,dim,depth,heads,mlp_dim,channels=512,dim_head=320,pool ='cls',emd_dropout=0.):
        super().__init__()
        image_height,image_width =pair(image_size)
        patch_height,patch_width =pair(patch_size)
        
        assert image_height%patch_height == 0 and image_width%patch_width==0,'Image dimension must be divisible by the patch size'
        num_patches =(image_height//patch_height) *(image_width//patch_width)
        patch_dim =channels*patch_height *patch_width  #128*5*5=3200
        
        assert pool in {'cls','mean'},'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding =nn.Sequential(
            Rearrange('b c (h p1)(w p2)-> b (h w) (p1 p2 c)', p1 =patch_height,p2 =patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim,dim),
            nn.LayerNorm(dim),
        )  #[24,128,40,100] ->[24,128,(8,5),(20,5)] ->[24,8,20,3200]

        self.pos_embedding = nn.Parameter(torch.randn(1,num_patches+1,dim))

        self.cls_token = nn.Parameter(torch.randn(1,1,dim))
        self.dropout =nn.Dropout(emd_dropout)

        self.transformer = Transformer(dim,depth,heads,dim_head,mlp_dim)

        self.pool =pool
        self.to_latent =nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            # nn.Linear(dim,num_classes)
        )

    def forward(self,img):
        # *_,h,w,dtype =*img.shape,img.dtype
        x = self.to_patch_embedding(img)
        # print(x.shape)
        b,n,_=x.shape

        cls_token =repeat(self.cls_token,'1 1 d -> b 1 d',b=b)
        # print(x.shape)
        x =torch.cat((cls_token,x),dim=1)
        # print(x.shape)
        x += self.pos_embedding[:,:(n+1)]
        # print(x.shape)
        mask = torch.ones_like(x) * 0.1
        x = x + torch.bernoulli(mask) * -1e12
        # print(x.shape)
        x =self.transformer(x)
        x =x.mean(dim = 1) if self.pool == 'mean' else x[:,0]
        x =self.to_latent(x)
        # print(self.mlp_head(x).shape)
        # print(self.mlp_head(x).flatten(1).shape)
        return self.mlp_head(x)
