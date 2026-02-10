import torch
import time

class ResonantDataset:
    def __init__(self, file_path, block_size=8):
        with open(file_path, 'r') as f:
            self.data = f.read()
        self.chars = sorted(list(set(self.data)))
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.encoded = torch.tensor([self.stoi[c] for c in self.data], dtype=torch.long)
        self.block_size = block_size

    def get_batch(self, batch_size):
        ix = torch.randint(len(self.encoded) - self.block_size, (batch_size,))
        x = torch.stack([self.encoded[i:i+self.block_size] for i in ix])
        y = torch.stack([self.encoded[i+1:i+self.block_size+1] for i in ix])
        return x, y, time.time() # This is the global_epoch for 39Hz sync

