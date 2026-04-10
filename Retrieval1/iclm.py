import torch
import torch.nn as nn
import torch.nn.functional as F

class ICLM(nn.Module):
    def __init__(self, embed_dim=128, heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.scale = (embed_dim // heads) ** -0.5

        self.v_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.a_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.a_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.a_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, v_o, a_o):
        B = v_o.size(0)

        def reshape_heads(x):
            return x.view(B, self.heads, 1, self.embed_dim // self.heads)

        vq = reshape_heads(self.v_q(v_o))
        vk = reshape_heads(self.v_k(v_o))
        vv = reshape_heads(self.v_v(v_o))

        aq = reshape_heads(self.a_q(a_o))
        ak = reshape_heads(self.a_k(a_o))
        av = reshape_heads(self.a_v(a_o))

        dots_v_ai = torch.einsum('bhid,bhjd->bhij', aq, vk) * self.scale
        attn_v_ai = dots_v_ai.softmax(dim=-1)
        v_ai = torch.einsum('bhij,bhjd->bhid', attn_v_ai, vv).reshape(B, self.embed_dim)

        dots_a_vi = torch.einsum('bhid,bhjd->bhij', vq, ak) * self.scale
        attn_a_vi = dots_a_vi.softmax(dim=-1)
        a_vi = torch.einsum('bhij,bhjd->bhid', attn_a_vi, av).reshape(B, self.embed_dim)

        v_local = torch.sigmoid(a_vi) * v_ai
        v_global = torch.sigmoid(a_o) * v_ai
        v_final = v_local + v_global + v_o

        a_fine = torch.sigmoid(v_ai) * a_o
        a_iteration = torch.sigmoid(a_vi) * a_o
        a_final = a_fine + a_iteration + a_o

        return F.normalize(v_final, p=2, dim=1), F.normalize(a_final, p=2, dim=1)