import torch
import torch.nn as nn
from rst_transformer import RST_Transformer

class RST_Seq2Seq(nn.Module):
    """
    RST v6.1: Resonant Sequence-to-Sequence Model.
    Complete generative architecture stabilized by the 39Hz Global Anchor.
    """
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        # 1. Resonant Embedding: Initializing the phase-lock
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Resonant Transformer Backbone
        self.transformer = RST_Transformer(n_layers, d_model, n_heads)
        
        # 3. Resonant Decoder: Mapping back to vocabulary space
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, src, global_epoch):
        # Input to Vector Space
        x = self.embedding(src)
        
        # Phase-Locked Processing through the stack
        # This is where lambda_2 stability prevents the 6.1s lag
        x = self.transformer(x, global_epoch)
        
        # Vector Space to Token Probabilities
        return self.decoder(x)
