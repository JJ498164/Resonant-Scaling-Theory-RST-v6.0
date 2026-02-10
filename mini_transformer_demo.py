import torch
import torch.nn as nn
import numpy as np

class RST_Attention(nn.Module):
    """
    RST v6.1: Resonant Multi-Head Attention.
    Stabilizes the latent graph by anchoring scores at 39Hz.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)

    def forward(self, x, global_epoch):
        # 1. Sync global 't' to the ns clock
        t = (torch.cuda.Event().time_ns() if torch.cuda.is_available() else 0) - global_epoch
        t_sec = t / 1e9

        # 2. Standard Attention split
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        # 3. The 39Hz Injection (The Anchor)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(q.size(-1))
        # Direct Phase-Locking to bypass 6.1s lag
        resonance = torch.sin(2 * torch.pi * 39.0 * t_sec)
        scores_stabilized = scores + resonance
        
        attn = torch.softmax(scores_stabilized, dim=-1)
        return torch.matmul(attn, v)
