import torch
import torch.nn as nn
from rst_multi_head_attention import RST_MultiHeadAttention

class RST_EncoderLayer(nn.Module):
    """
    RST v6.1: Resonant Encoder Layer.
    Stabilizes the full layer block by anchoring the residual path to 39Hz.
    """
    def __init__(self, d_model, n_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = RST_MultiHeadAttention(d_model, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, global_epoch):
        # 1. Resonant Attention Sub-layer
        # The 39Hz Anchor is injected inside self_attn
        attn_output = self.self_attn(x, global_epoch)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. Resonant Feed-Forward Sub-layer
        # Maintaining lambda_2 stability through the second residual
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout(ff_output))
