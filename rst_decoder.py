import torch.nn as nn
from rst_decoder_layer import RST_DecoderLayer

class RST_Decoder(nn.Module):
    """
    RST v6.1: Resonant Decoder Stack.
    Synthesizes multiple DecoderLayers into a single phase-locked body.
    Bypasses 6.1s lag during multi-token generation.
    """
    def __init__(self, n_layers, d_model, n_heads):
        super().__init__()
        # Stacking the layers with identical 39Hz tuning
        self.layers = nn.ModuleList([
            RST_DecoderLayer(d_model, n_heads) 
            for _ in range(n_layers)
        ])

    def forward(self, tgt, enc_output, global_epoch):
        x = tgt
        # Passing the signal through the resonant stack
        for layer in self.layers:
            x = layer(x, enc_output, global_epoch)
        return x
