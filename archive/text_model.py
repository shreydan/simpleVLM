import torch
import torch.nn as nn
import torch.nn.functional as F
from kv_cache import KVCache

class LlamaRMSNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(
            torch.ones(self.embed_dim,dtype=torch.float32),
            requires_grad=True
        )
        
    def forward(self, x):
        # x [B, S, D]
        mean = x.pow(2).mean(dim=-1,keepdim=True)
        r_sqrt = x * torch.rsqrt(mean + 1e-5) # [B, S, 1]
        y = r_sqrt * self.weight
        return y.to(x.dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # x [B S D]
        return x * F.sigmoid(x)


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.intermediate_dim = config.intermediate_dim
        self.gate_proj = nn.Linear(self.embed_dim, self.intermediate_dim, bias=False, dtype=config.dtype)
        self.up_proj = nn.Linear(self.embed_dim, self.intermediate_dim, bias=False, dtype=config.dtype)
        self.down_proj = nn.Linear(self.intermediate_dim, self.embed_dim, bias=False, dtype=config.dtype)
        self.act_fn = SiLU()
        
    def forward(self, x):
        # x [B S D]
        x1 = self.gate_proj(x)
        x2 = self.up_proj(x)
        x = self.act_fn(x1) * x2
        x = self.down_proj(x)
        return x


def precompute_rope(head_dim, base_theta=10_000, context_length=4096):
    k = torch.arange(0,head_dim,2)[:head_dim//2].float()
    inv_freq = 1 / (base_theta ** (k/head_dim))

    positions = torch.arange(context_length)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0) # [S, H/2]
    angles = torch.cat([angles, angles],dim=-1) # [S, H]

    cos = torch.cos(angles) # [S, H]
    sin = torch.sin(angles) # [S, H]


    return cos, sin

def apply_rope(x, cos, sin, positions=None):
    B, nH, S, H = x.shape
    x1 = x[...,:H//2] # [B, nH, S, H/2]
    x2 = x[...,H//2:] # [B, nH, S, H/2]
    if positions is None:
        positions = torch.arange(S)
    cos_values = cos[positions,:].unsqueeze(0).unsqueeze(1) # [1,1,S,H]
    sin_values = sin[positions,:].unsqueeze(0).unsqueeze(1) # [1,1,S,H]
    rotated = torch.cat([-x2,x1],dim=-1)
    x_rope = (x * cos_values) + (rotated * sin_values)
    return x_rope.to(x.dtype)



class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed_dim = config.embed_dim
        self.num_kv_heads = config.num_kv_heads
        self.num_q_heads = config.num_q_heads
        self.max_position_embeddings = config.max_position_embeddings

        assert self.embed_dim % self.num_q_heads == 0, "embed_dim must be divisible by num_q_heads"
        assert self.num_q_heads % self.num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"

        self.head_dim = self.embed_dim // self.num_q_heads

        self.q_proj = nn.Linear(self.embed_dim, self.head_dim * self.num_q_heads, bias=False, dtype=config.dtype)
        self.k_proj = nn.Linear(self.embed_dim, self.head_dim * self.num_kv_heads, bias=False, dtype=config.dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.head_dim * self.num_kv_heads, bias=False, dtype=config.dtype)

        self.drop = nn.Dropout(config.attn_dropout)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False, dtype=config.dtype)

        cos, sin = precompute_rope(self.head_dim, base_theta=config.base_theta,  # Important: RoPE applies to half dimension
                                  context_length=self.max_position_embeddings)

        self.register_buffer('mask',
                             torch.triu(torch.ones(
                                 config.max_position_embeddings,
                                 config.max_position_embeddings
                             ),diagonal=1))
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.kv_cache = None
        self.use_cache = False

    def enable_kv_cache(self, dtype=None):
        self.kv_cache = KVCache(self.max_position_embeddings, self.head_dim, self.num_kv_heads, dtype, self.rope_cos.device)
        self.use_cache = True
    
    def reset_kv_cache(self):
        if self.kv_cache is not None:
            self.kv_cache.reset()

    def forward(self, x, attention_mask):
        # x: [B, S, D]
        B, S, D = x.shape

        q = self.q_proj(x) # [B S H*nQ]
        k = self.k_proj(x) # [B S H*nKV]
        v = self.v_proj(x) # [B S H*nKV]

        q = q.view(B, S, self.num_q_heads, self.head_dim).transpose(1,2) # [B nQ S H]
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1,2) # [B nKV S H]
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1,2) # [B nKV S H]

        if self.use_cache and self.kv_cache is not None:
            assert B == 1, "Batch size must be 1 in inference when using KV cache."
            past_length = self.kv_cache.length
            positions = torch.arange(past_length, past_length + S, device=x.device)
        else:
            positions = torch.arange(0, S, device=x.device)

        # Apply RoPE
        q = apply_rope(q, self.rope_cos, self.rope_sin, positions)
        k = apply_rope(k, self.rope_cos, self.rope_sin, positions)

        if self.use_cache and self.kv_cache is not None:
            self.kv_cache.update(k, v)
            k, v = self.kv_cache.get()
            total_length = k.shape[2]
        else:
            total_length = S

        k = k.repeat_interleave(self.num_q_heads//self.num_kv_heads, dim=1) # [B nQ S H]
        v = v.repeat_interleave(self.num_q_heads//self.num_kv_heads, dim=1) # [B nQ S H]

        attn = q @ k.transpose(2,3) # [B nQ S1 H] @ [B nQ H S2] = [B nQ S1 S2]
        
        if self.use_cache and self.kv_cache is not None:
            mask = self.mask[past_length:past_length+S, :total_length].bool()
        else:
            mask = self.mask[:S, :S].bool()

        attn.masked_fill_(mask, -torch.inf)
        
        attn = F.softmax(attn / (self.head_dim ** 0.5), dim=-1)

        attn = self.drop(attn)


        out = attn @ v # [B nQ S S] @ [B nQ S H] = [B nQ S H]
        out = out.transpose(1,2) # [B S nQ H]
        out = out.reshape(B, S, D)

        proj = self.o_proj(out)
        
        return proj


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.self_attn = GroupedQueryAttention(config)
        self.mlp = LlamaMLP(config)
        
        self.input_layernorm = LlamaRMSNorm(config.embed_dim)
        self.post_attention_layernorm = LlamaRMSNorm(config.embed_dim)
        
        
    def forward(self, x, attention_mask):
        # x [B S D]
        skip = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = x + skip
        
        skip = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + skip
        
        return x


class LLaMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            self.config.vocab_size, 
            self.config.embed_dim, 
            padding_idx=self.config.eos_token_id,
            dtype=self.config.dtype)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(self.config) for _ in range(self.config.num_layers)
        ])

        self.norm = LlamaRMSNorm(self.config.embed_dim)
        self.lm_head = nn.Linear(self.config.embed_dim, self.config.vocab_size, bias=False, dtype=self.config.dtype)

        self._tie_weights()

    def _tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def enable_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.enable_kv_cache(dtype=self.config.dtype)

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.kv_cache.reset()


if __name__ == '__main__':
    from types import SimpleNamespace
    config = SimpleNamespace(
        embed_dim = 576,
        intermediate_dim = 1536,
        max_position_embeddings = 8192,
        base_theta = 100000,
        num_q_heads = 9,
        num_kv_heads = 3,
        attn_dropout = 0.,
        num_layers = 30,
        vocab_size = 49152,
        dtype = torch.bfloat16,
        eos_token_id = 2
        )
    model = LLaMA(config)
    x = torch.randint(0,config.vocab_size,(1,1)).long()
    outputs = model(x)
    print(outputs.shape)