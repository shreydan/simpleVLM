import torch
import torch.nn as nn
from types import SimpleNamespace
from llama import LLaMA

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
    eos_token_id = 2,
    dtype = torch.float32,
)

model = LLaMA(config)

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('./simpleVLM')
smol = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

smol_sd = smol.state_dict()
model_sd = model.state_dict()
smol_sd = {k:v for k,v in smol_sd.items() if not any([s in k for s in ['rope','causal_mask']])}
# model_sd = {k:v for k,v in model_sd.items() if not any([s in k for s in ['rope','causal_mask']])}

# for k1,k2 in zip(smol_sd,model_sd):
#     print(k1, k2, k1.replace('model.','')==k2)

for smol_key,smol_value in smol_sd.items():
    model_key = smol_key.replace('model.','')
    model_sd[model_key] = smol_value.clone()

model.load_state_dict(model_sd)


print('params',sum([p.numel() for p in smol.parameters()]),sum([p.numel() for p in model.parameters()]))
print('verify',torch.allclose(smol.model.embed_tokens.weight, model.embed_tokens.weight))

# check weight ties

print('tied weights',torch.allclose(model.embed_tokens.weight, model.lm_head.weight))

def update_embeddings(model, old_vocab_size, new_vocab_size, dim):
    old_dim = model.embed_tokens.weight.shape[0]
    assert old_vocab_size == old_dim, 'what'
    new_embedding = nn.Embedding(new_vocab_size,dim)
    print('new_embedding_weight',new_embedding.weight.shape)
    new_embedding.weight.data[:old_vocab_size] = model.embed_tokens.weight.data.clone()
    # this is important, random init is ruining the outputs before training. anyway all will be masked to -100
    # to check after training if new tokens still have zero weights or not!
    nn.init.zeros_(new_embedding.weight.data[old_vocab_size:])
    model.embed_tokens = new_embedding
    model.lm_head = nn.Linear(dim, new_vocab_size)
    model._tie_weights()
    return model

model = update_embeddings(model,config.vocab_size,config.vocab_size+3,config.embed_dim)

print('new embeddings',model.embed_tokens.weight.shape, 'new lm_head',model.lm_head.weight.shape)
print('tied weights NEW!',torch.allclose(model.embed_tokens.weight, model.lm_head.weight))

random_idxs = torch.randint(0,model.embed_tokens.weight.shape[0],(5,)).long()
for idx in random_idxs:
    _idx_tensor = torch.tensor([idx]).long()
    print(idx, torch.allclose(model.embed_tokens(_idx_tensor),smol.model.embed_tokens(_idx_tensor)))

special_tokens = [2,49152,49153,49154]
for idx in special_tokens:
    _idx_tensor = torch.tensor([idx]).long()
    print(idx, tokenizer.decode([idx]),model.embed_tokens(_idx_tensor).shape)


def get_input(text):
    messages = [{"role": "user", "content": text}]
    input_text=tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    return inputs

def generate(
    model,
    input_ids,
    max_new_tokens=32,
    context_length = config.max_position_embeddings,
    temperature = 0.,
    eos_token_id = config.eos_token_id
):
    model.eval()
    inputs = input_ids.clone()
    # print(tokenizer.decode(inputs.flatten().numpy()))
    for _ in range(max_new_tokens):
        context = inputs[:,-context_length:]
        with torch.inference_mode():
            logits = model(context)
            logits = logits[:,-1,:]

            if temperature > 0.:
                logits = logits / temperature

            probs = logits.softmax(dim=-1)

            if temperature > 0.:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs,dim=-1,keepdim=True)

            if next_token == eos_token_id:
                break
        # print(tokenizer.decode(next_token.flatten().numpy()),end='')
        inputs = torch.cat([inputs, next_token],dim=1)  
    # print()
    return inputs            

print('checking if we didnt mess up increasing the embeddings size:\n')

inputs = get_input('give me a random fact about llamas')
# print(inputs)
print('generated...')
generated = generate(model, inputs, max_new_tokens=80, temperature=0.125)
print(tokenizer.decode(generated.flatten().numpy(),skip_special_tokens=False))