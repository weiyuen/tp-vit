"""
ViT model with TPR Block and TPR Inference Block added on top.
ViT based on lucidrains implementation.
"""

import torch

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


# helper
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# --- lucidrains base classes ---
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    

# --- My TPR-related classes ---
class TPR(nn.Module):
    def __init__(self, seq_length, n_roles, dim, heads, dim_head, dropout):
        super().__init__()
        self.role = nn.Linear(seq_length, n_roles)
        self.filler = Attention(
            dim=dim,
            heads=heads, 
            dim_head=dim_head, 
            dropout=dropout
        )
    
    def forward(self, x):
        # In: b n d
        f = self.filler(x) # Out: b n d
        
        x = rearrange(x, 'b n d -> b d n')
        r = self.role(x) # Out: b d r
        
        # Reshape to prepare for multiplication
        r = torch.unsqueeze(r, dim=2) # Out: b d 1 r
        f = rearrange(f, 'b n d -> b d n 1')
        tpr = torch.matmul(f, r) # Out: b d n r
        tpr = rearrange(tpr, 'b d n r -> b n r d')
        return tpr
        

class TPRInferenceAttention(nn.Module):
    '''
    Turns TPR of shape b n r d into output of shape b n d
    by using TPR as the query to transformer output token,
    with one role per attention head
    '''
    def __init__(self, dim, n_roles=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = n_roles # one att. head per role
        inner_dim = dim_head * self.heads
        project_out = not (self.heads==1 and dim_head==dim)

        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_q = nn.Linear(dim, dim_head, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    def forward(self, tpr, embedding):
        # tpr dim: b n r d
        # embedding dim: b n d
        q = self.to_q(tpr)
        kv = self.to_kv(embedding).chunk(2, dim = -1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        
        # use tpr as query
        q = rearrange(q, 'b n r d -> b r n d') # r = h

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    
class TPRPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, tpr, embedding):
        return self.fn(tpr, self.norm(embedding)) # Check embedding logic
    

class TPRInferenceBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TPRPreNorm(dim, TPRInferenceAttention(dim, n_roles = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, tpr, embedding):
        for attn, ff in self.layers:
            x = attn(tpr, embedding) + embedding # Check skip connection logic
            x = ff(x) + x
        return x
    
    
class TPViT(nn.Module):
    def __init__(
        self,
        *, 
        image_size, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        heads, 
        mlp_dim, 
        pool='cls', 
        channels=3, 
        dim_head=64, 
        dropout=0., 
        emb_dropout=0.,
        n_roles=8,
        tpr_depth=1
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        # self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.tpr_block = TPR(
            seq_length=num_patches+1, 
            n_roles=n_roles,
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        
        self.tpr_inference_block = TPRInferenceBlock(dim, tpr_depth, heads, dim_head, mlp_dim, dropout)
        
        # self.temp_head = nn.Linear(n_roles, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        
        tpr = self.tpr_block(x) # Out: b n r d
        
        # Use TPRInferenceBlock
        x = self.tpr_inference_block(tpr, x) # Out: b n d
        
        # Mean pooling
        x = x.mean(dim=1) # Out: b d
        
        '''
        # Temporary code to get working output
        x = rearrange(x, 'b n r d -> b r (n d)')
        x = x.mean(dim=2) # Out: b r 1
        x = torch.squeeze(x)
        x = self.temp_head(x)
        '''
        
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        # x = self.to_latent(x)
        
        return self.mlp_head(x)

