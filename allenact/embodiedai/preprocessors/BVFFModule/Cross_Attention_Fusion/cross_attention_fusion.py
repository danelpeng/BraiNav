import torch 
import torch.nn as nn
from torch import einsum

from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads
        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)
        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class cross_attn_fusion(nn.Module):
    def __init__(self, cross_attn_depth=1):
        super().__init__()
        self.pool = 'cls'
        bf_dim, vf_dim = 768, 1568 # token: 768, 1568 fMRI：1400, 1568
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(bf_dim, vf_dim),
                nn.Linear(vf_dim, bf_dim),
                PreNorm(vf_dim, CrossAttention(vf_dim, heads=3)),
                nn.Linear(vf_dim, bf_dim),
                nn.Linear(bf_dim, vf_dim),
                PreNorm(bf_dim, CrossAttention(bf_dim, heads=3)),
            ]))
    
    def forward(self, bf, vf):
           
        for f_bv, g_vb, cross_attn_b, f_vb, g_bv, cross_attn_v in self.cross_attn_layers:
            # cross attn for visual feat
            cal_q = f_vb(vf)
            cal_qkv = torch.cat((cal_q, bf), dim=1)
            cal_out = cal_q + cross_attn_v(cal_qkv)
            cal_out_1 = g_bv(cal_out)
            vf = cal_out_1
            # cross attn for brain feat 
            cal_q = f_bv(bf)
            cal_qkv = torch.cat((cal_q, vf), dim=1)
            cal_out = cal_q + cross_attn_b(cal_qkv)
            cal_out_2 = g_vb(cal_out)
            bf = cal_out_2
        
        if(self.pool=='cls'):
            return torch.cat([vf, bf], dim=-1)


# small: b   large: v

class FiLM_fusion(nn.Module):
    def __init__(self, visual_dim = 1568, cond_dim = 768):
        super(FiLM_fusion, self).__init__()
        # FiLM generator: from neural_embed to (gamma, beta)
        self.film_gen = nn.Linear(cond_dim, 2 * visual_dim)
    
    def forward(self, neural_embed, visual_embed):
        """
        visual_embed: (B, 1, Dv)
        neural_embed: (B, 1, Dn)
        """
        B, _, Dv = visual_embed.shape
        _, _, Dn = neural_embed.shape

        # Remove singleton dimension for FiLM generator
        neural_embed_flat = neural_embed.squeeze(1)  # (B, Dn)
        gamma_beta = self.film_gen(neural_embed_flat)  # (B, 2*Dv)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)  # Each is (B, Dv)

        # Unsqueeze to match visual_embed shape
        gamma = gamma.unsqueeze(1)  # (B, 1, Dv)
        beta = beta.unsqueeze(1)    # (B, 1, Dv)

        # Apply FiLM
        modulated = gamma * visual_embed + beta  # (B, 1, Dv)

        return modulated