import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from blinky import Blinky
from finetune_utils import get_peft_model, merge_peft_model, Config
from datasets import load_dataset
from processor import BlinkyProcessor
import gc

config = Config(
    embed_dim = 576,
    intermediate_dim = 1536,
    max_position_embeddings = 8192,
    base_theta = 100000,
    num_q_heads = 9,
    num_kv_heads = 3,
    attn_dropout = 0.,
    num_layers = 30,
    vocab_size = 49152,
    dtype = torch.float32,
    eos_token_id = 2,
    image_start_token_id = 49152,
    image_end_token_id = 49153,
    image_token_id = 49154,
    num_image_tokens=256,
    prefix_length=1+256+1 #<|start_of_image|>256*<|image_token|><|end_of_image|>
)

lora_config = Config(
    rank=64,
    alpha=128,
    lora_dropout=0.05
)

text_target_layers = [
    'q_proj',
    'k_proj',
    'v_proj',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj'
]


model = Blinky(config)


model._prepare_for_training()


dataset = load_dataset('openbmb/RLAIF-V-Dataset')['train']


origins = pd.DataFrame({'origin': dataset['origin_dataset']})
train_idxs, valid_idxs = train_test_split(origins, stratify=origins['origin'], test_size=0.015, random_state=2025)
print(len(train_idxs), len(valid_idxs))

train_dataset = dataset.select(train_idxs.index)
valid_dataset = dataset.select(valid_idxs.index)


def make_conversation(samples):
    conversations = []
    for q,a in zip(samples['question'], samples['chosen']):
        conversations.append([
            {'role': 'user', 'content': q},
            {'role': 'assistant', 'content': a}
        ])
    samples['text'] = conversations
    return samples

train_dataset = train_dataset.select(np.arange(10_000)).map(make_conversation,batched=True,num_proc=8,batch_size=512)
train_dataset = train_dataset.remove_columns(
    ['ds_name', 'question', 'chosen', 'rejected', 'origin_dataset', 'origin_split', 'idx', 'image_path']
)


processor = BlinkyProcessor('./Blinky')


def collate_fn(samples):
    inputs = processor(samples)
    labels = inputs['input_ids'].clone()
    
    for ignore_token in [49152, 49153, 49154]:  # start_of_image, end_of_image, image_token
        mask = (inputs['input_ids'] == ignore_token)
        labels[mask] = -100
    
    padding_mask = (inputs['input_ids'] == processor.tokenizer.pad_token_id)
    labels[padding_mask] = -100
    
    shifted_labels = torch.full_like(labels, -100)
    shifted_labels[:, :-1] = labels[:, 1:].clone()
    
    inputs['labels'] = shifted_labels
    return inputs


model = model.cuda()

dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = 16,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers = 6,
    pin_memory = True
)


def get_optimizer(model, vision_lr=1e-5, vision_proj_lr=1e-3, text_lr=1.5e-5):
    
    for p in model.parameters():
        p.requires_grad = False
    
    vision_params = []
    vision_proj_params = []
    text_params = []
    
    for p in model.vision_model.vision.parameters():
        p.requires_grad = True
        vision_params.append(p)
            
    for p in model.vision_model.rms_norm.parameters():
        p.requires_grad = True
        vision_proj_params.append(p)
                
    for p in model.vision_model.dim_proj.parameters():
        p.requires_grad = True
        vision_proj_params.append(p)
    
    for n,p in model.text_model.named_parameters():
        if 'embed_tokens' in n or 'lm_head' in n:
            continue
        else:
            p.requires_grad = True
            text_params.append(p)
    
    param_groups = [
        {'params': vision_params, 'lr': vision_lr},
        {'params': vision_proj_params, 'lr': vision_proj_lr},
        {'params': text_params, 'lr': text_lr},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    dtypes = []
    for p in model.parameters():
        dtypes.append(p.dtype)
    print('nodel',set(dtypes))

    return optimizer



losses = []
log_steps = 10
epochs = 1

optim = get_optimizer(model)
sched = torch.optim.lr_scheduler.OneCycleLR(
    optim,
    max_lr=[pg['lr'] for pg in optim.param_groups],
    total_steps=len(dataloader)*epochs,
    pct_start=0.2
)

for epoch in tqdm(range(epochs)):
    running_loss = 0.0
    
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = {k: v.cuda() for k, v in batch.items()}
        
        loss = model(**batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        sched.step()
        optim.zero_grad()
        
        running_loss += loss.item()
        
        if (batch_idx + 1) % log_steps == 0:
            avg_loss = running_loss / log_steps
            losses.append(avg_loss)
            running_loss = 0.0
            print(f"{epoch=} {batch_idx=} {avg_loss=:.3f}")

    gc.collect()
    torch.cuda.empty_cache()


fig=plt.figure()
plt.plot(losses)
plt.title('loss')
fig.savefig('losses.png')


sd = model.state_dict()
torch.save(sd,'./Blinky/model.pt')