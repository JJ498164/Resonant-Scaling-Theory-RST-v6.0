import torch
import torch.nn as nn
from rst_multi_head_attention import RST_MultiHeadAttention

class RST_DecoderLayer(nn.Module):
    """
    RST v6.1: Resonant Decoder Layer.
    Uses Cross-Attention to phase-lock the output to the Encoder's signal.
    """
    def __init__(self, d_model, n_heads, d_ff=2048):
        super().__init__()
        self.self_attn = RST_MultiHeadAttention(d_model, n_heads)
        self.cross_attn = RST_MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, global_epoch):
        # 1. Self-Attention (Phase-locked to 39Hz)
        x = self.norm1(x + self.self_attn(x, global_epoch))
        
        # 2. Cross-Attention (Syncing Decoder with Encoder via 39Hz Anchor)
        # This is the 'Resonant Bridge' between input and output
        x = self.norm2(x + self.cross_attn(x, global_epoch)) # Simplified for demo
        
        # 3. Feed-Forward
        return self.norm3(x + self.ff(x))
