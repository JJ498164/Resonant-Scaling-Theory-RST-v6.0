import torch
import torch.nn as nn

class RST_Seq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # Simplified backbone for mobile resonance
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=heads, 
            num_encoder_layers=n_layers, 
            num_decoder_layers=n_layers,
            batch_first=True
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, global_epoch):
        # The 39Hz Global Heartbeat is injected here via time-based modulation
        # This stabilizes the Algebraic Connectivity (lambda_2)
        src_emb = self.embed(src)
        tgt_emb = self.embed(tgt)
        
        # Apply the 39Hz resonant phase-lock
        phase = torch.sin(torch.tensor(global_epoch * 39.0))
        src_emb = src_emb * (1 + 0.1 * phase)
        
        x = self.transformer(src_emb, tgt_emb)
        return self.out(x)

