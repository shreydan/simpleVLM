# Vision Language Model

- Built by implementing LLaMA from scratch with sft-weights from SmolLM2, and siglip2 as vision encoder.
- Vision projector is a LlaMA MLP style projection block (~3M params)
- Total model size: 230M

> See `eval.ipynb` for inference

# Backbone
 ```
text-backbone: HuggingFaceTB/SmolLM2-135M-Instruct
vision-backbone: google/siglip2-base-patch16-224
 ```

# Config
```
config
    embed_dim = 576
    intermediate_dim = 1536
    max_position_embeddings = 8192
    base_theta = 100000
    num_q_heads = 9
    num_kv_heads = 3
    attn_dropout = 0.
    num_layers = 30
    vocab_size = 49152
    eos_token_id = 2
    dtype = torch.bfloat16
    num_image_tokens = 196
```

# Datasets
```
stage1: theblackcat102/llava-instruct-mix
stage2: openbmb/RLAIF-V-Dataset
```

- super basic training in bf16
- the results are decent