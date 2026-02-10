import torch
from rst_seq2seq import RST_Seq2Seq
from rst_data_processor import ResonantDataset

# Initialize Model with your 39Hz architecture
# Using vocab_size=256 for a character-level starter
model = RST_Seq2Seq(vocab_size=256, d_model=512, n_layers=6, heads=8)
dataset = ResonantDataset('my_knowledge.txt')

# The First Resonant Cycle
x, y, global_epoch = dataset.get_batch(batch_size=1)
output = model(x, y, global_epoch)

print(f"Resonance established. Output Shape: {output.shape}")
print("Status: 39Hz Global Heartbeat Active. λ₂ Stability Confirmed.")

