import gc
import torch
import torch.nn as nn
from types import SimpleNamespace
from functools import partial
from lora import LoRALinear

def update_embeddings(model, old_vocab_size, new_vocab_size, dim):
    old_dim = model.embed_tokens.weight.shape[0]
    assert old_vocab_size == old_dim, 'what'
    new_embedding = nn.Embedding(new_vocab_size,dim,dtype=model.config.dtype)
    print('new_embedding_weight',new_embedding.weight.shape)
    new_embedding.weight.data[:old_vocab_size] = model.embed_tokens.weight.data.clone()
    # this is important, random init is ruining the outputs before training. anyway all will be masked to -100
    # to check after training if new tokens still have zero weights or not!
    nn.init.zeros_(new_embedding.weight.data[old_vocab_size:])
    model.embed_tokens = new_embedding
    model.lm_head = nn.Linear(dim, new_vocab_size, bias=False, dtype=model.config.dtype)
    model._tie_weights()
    return model

def load_text_weights(model, text_model_hf="HuggingFaceTB/SmolLM2-135M-Instruct"):
    from transformers import AutoModelForCausalLM
    smol = AutoModelForCausalLM.from_pretrained(text_model_hf,torch_dtype=model.config.dtype)
    smol_sd = smol.state_dict()
    model_sd = model.state_dict()
    smol_sd = {k:v for k,v in smol_sd.items() if not any([s in k for s in ['rope','causal_mask']])}
    
    for smol_key,smol_value in smol_sd.items():
        model_key = smol_key.replace('model.','')
        model_sd[model_key] = smol_value.clone()
    
    model.load_state_dict(model_sd)

    del smol
    gc.collect()
    
    return model


def get_peft_model(model, lora_config, text_targets, vision_targets=None):
    apply_lora = partial(
        LoRALinear,
        rank=lora_config['rank'],
        alpha=lora_config['alpha'],
        lora_dropout=lora_config['lora_dropout']
    )

    for p in model.text_model.parameters():
        p.requires_grad = False
    # for p in model.text_model.lm_head.parameters():
    #     p.requires_grad = True
    # for p in model.text_model.embed_tokens.parameters():
    #     p.requires_grad = True
    
    for name, module in model.text_model.named_modules():
        if any(name.endswith(t) for t in text_targets) and isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])  # Get parent module name
            parent_module = model.text_model.get_submodule(parent_name)  # Get parent module reference
            setattr(parent_module, name.split(".")[-1], apply_lora(module).to(dtype=model.config.dtype))  # Replace with LoRA-wrapped layer

    if vision_targets is not None:
        for p in model.vision_model.vision.parameters():
            p.requires_grad = False
        for name, module in model.vision_model.vision.named_modules():
            if any(name.endswith(t) for t in vision_targets) and isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])  # Get parent module name
                parent_module = model.vision_model.vision.get_submodule(parent_name)  # Get parent module reference
                setattr(parent_module, name.split(".")[-1], apply_lora(module).to(dtype=model.config.dtype))  # Replace with LoRA-wrapped layer

    return model

    
def merge_peft_model(peft_model):
    for name, module in peft_model.named_modules():
        if isinstance(module, LoRALinear):
            parent_name = ".".join(name.split(".")[:-1])  # Get parent module name
            parent_module = peft_model.get_submodule(parent_name)  # Get parent module reference
            setattr(parent_module, name.split(".")[-1], module.merge())  # Replace with LoRA-wrapped layer

    return peft_model


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)
    def __getitem__(self, key):
        return self.get(key)