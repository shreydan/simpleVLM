import torch

class KVCache:
    def __init__(self, max_length, head_dim, n_heads, dtype=torch.float32, device='cpu'):
        self.max_length = max_length
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.device = device
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.keys = torch.zeros((1, self.n_heads, self.max_length, self.head_dim),
                                device=self.device, dtype=self.dtype)
        self.values = torch.zeros((1, self.n_heads, self.max_length, self.head_dim),
                                  device=self.device, dtype=self.dtype)
        self.length = 0

    def update(self, new_key, new_value):
        # new_key/new_value: [B, n_heads, S, head_dim] (with B==1 during inference)
        S = new_key.shape[2]
        assert self.length + S <= self.max_length, "KV cache overflow"
        seq_start = self.length
        seq_end = seq_start + S
        self.keys[:, :, seq_start:seq_end, :] = new_key
        self.values[:, :, seq_start:seq_end, :] = new_value
        self.length = seq_end

    def get(self):
        if self.length == 0:
            return None, None
        return self.keys[:, :, :self.length, :], self.values[:, :, :self.length, :]