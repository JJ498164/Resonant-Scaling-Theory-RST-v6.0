import torch
import torch.nn as nn
from rst_encoder_layer import RST_EncoderLayer

class RST_Transformer(nn.Module):
    """
    RST v6.1: Full Resonant Transformer.
    Synthesizes Multi-Head Attention and Encoder Layers into a 
    global phase-locked architecture to eliminate 6.1s lag.
    """
    def __init__(self, n_layers, d_model, n_heads, d_ff=2048):
        super().__init__()
        # Creating the stack of resonant layers
        self.layers = nn.ModuleList([
            RST_EncoderLayer(d_model, n_heads, d_ff) 
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, global_epoch):
        # Every layer in the stack references the same 39Hz global heartbeat
        for layer in self.layers:
            x = layer(x, global_epoch)
        
        # Final stabilization of the output vector
        return self.final_norm(x)
