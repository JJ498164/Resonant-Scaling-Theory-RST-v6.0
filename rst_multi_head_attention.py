import torch
import torch.nn as nn
import math

class RST_MultiHeadAttention(nn.Module):
    """
    RST v6.1: Multi-Head Resonant Attention.
    Synchronizes parallel attention heads to a global 39Hz anchor.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, global_epoch):
        batch, seq_len, d_model = x.size()
        
        # 1. Global Clock Sync
        t = (torch.cuda.Event().time_ns() if torch.cuda.is_available() else 0) - global_epoch
        t_sec = t / 1e9

        # 2. Linear Projections and Head Splitting
        q = self.q_linear(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Scaled Dot-Product Attention with 39Hz Anchor
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Injection: Every head vibrates at 39Hz to maintain λ2 stability
        resonance = torch.sin(2 * torch.pi * 39.0 * t_sec)
        scores_stabilized = scores + resonance
        
        attn = torch.softmax(scores_stabilized, dim=-1)
        context = torch.matmul(attn, v)

        # 4. Merge Heads and Final Projection
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(context)
