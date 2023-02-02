"""
ViT model with TPR Block and TPR Inference Block added on top.
ViT based on huggingface implementation.
"""

import torch

from einops import rearrange
from torch import nn
from transformers import ViTModel


# --- lucidrains base classes (used in my TPR block) ---
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    def __init__(self, dim, heads=8, dim_head=96, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
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
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


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
        f = self.filler(x)  # Out: b n d

        x = rearrange(x, 'b n d -> b d n')
        r = self.role(x)  # Out: b d r

        # Reshape to prepare for multiplication
        r = torch.unsqueeze(r, dim=2)  # Out: b d 1 r
        f = rearrange(f, 'b n d -> b d n 1')
        tpr = torch.matmul(f, r)  # Out: b d n r
        tpr = rearrange(tpr, 'b d n r -> b n r d')
        return tpr


class TPRInferenceAttention(nn.Module):
    '''
    Turns TPR of shape b n r d into output of shape b n d
    by using TPR as the query to transformer output token,
    with one role per attention head
    '''
    def __init__(self, dim, tpr_dim_head=96, n_roles=8, dropout=0.):
        super().__init__()
        self.heads = n_roles # one att. head per role
        inner_dim = tpr_dim_head * self.heads
        project_out = not (self.heads==1 and tpr_dim_head==dim)

        self.scale = tpr_dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, tpr_dim_head, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, tpr, embedding):
        # tpr dim: b n r d
        # embedding dim: b n d
        q = self.to_q(tpr)
        kv = self.to_kv(embedding).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        # use tpr as query
        q = rearrange(q, 'b n r d -> b r n d')  # r = h

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
        return self.fn(tpr, self.norm(embedding))  # Check embedding logic


class TPRInferenceBlock(nn.Module):
    def __init__(self, dim, tpr_dim_head, depth, n_roles, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TPRPreNorm(dim, TPRInferenceAttention(dim, tpr_dim_head=tpr_dim_head, n_roles=n_roles, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, tpr, embedding):
        for attn, ff in self.layers:
            x = attn(tpr, embedding) + embedding  # Check skip connection logic
            x = ff(x) + x
        return x


# --- My TPR blocks stacked on HF ViT (no MLP head) ---
class TPViT(nn.Module):
    def __init__(
            self,
            dim=768,
            mlp_dim=3072,
            heads=12,
            dim_head=64,
            dropout=.1,
            n_roles=12,
            tpr_dim_head=64,
            tpr_depth=1,
            freeze_encoder=True
    ):
        super().__init__()

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        if freeze_encoder:
            for name, param in self.vit.named_parameters():
                param.requires_grad = False

        self.tpr_block = TPR(
            seq_length=197,
            n_roles=n_roles,
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )

        self.tpr_inference_block = TPRInferenceBlock(dim, tpr_dim_head, tpr_depth, n_roles, mlp_dim, dropout)

    def forward(self, x):
        x = self.vit(x)
        emb = x.last_hidden_state  # b, n, d

        # TPR and TPRInferenceBlock
        tpr = self.tpr_block(emb)  # Out: b n r d
        x = self.tpr_inference_block(tpr, emb)  # Out: b n d

        # Mean pooling
        x = x.mean(dim=1)  # Out: b d

        return x
