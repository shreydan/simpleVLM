import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from blinky import Blinky
from datasets import load_dataset
from processor import BlinkyProcessor
import gc
from types import SimpleNamespace
import ast

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
    dtype = torch.bfloat16,
    num_image_tokens = 196,
    vision_model_hf = 'google/siglip2-base-patch16-224',
    text_model_hf = 'HuggingFaceTB/SmolLM2-135M-Instruct'
)

model = Blinky(config)
model.prepare_for_training()

stage1_sd = torch.load('./Blinky/stage1.pt',weights_only=True)
model.load_state_dict(stage1_sd)

# RLAIF-V-Dataset

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

train_dataset = train_dataset.map(make_conversation,batched=True,num_proc=8,batch_size=512)
train_dataset = train_dataset.remove_columns(
    ['ds_name', 'question', 'chosen', 'rejected', 'origin_dataset', 'origin_split', 'idx', 'image_path']
)

# def preprocess_llava(samples):
#     conversations = samples['conversations']
#     conversations = [ast.literal_eval(c.replace('<image>','').strip()) for c in conversations]
#     updated_conversations = []
#     for conversation in conversations:
#         updated_conversation = [{
#             'role': 'user' if c['from']=='human' else 'assistant',
#             'content': c['value'].strip()
#         } for c in conversation]
#         updated_conversations.append(updated_conversation)
#     samples['conversations'] = updated_conversations
#     return samples

# dataset = load_dataset("theblackcat102/llava-instruct-mix")['train']
# dataset = dataset.map(preprocess_llava,batched=True,num_proc=8,batch_size=512)
# train_dataset = dataset.rename_column('conversations','text')

processor = BlinkyProcessor('./Blinky')

def collate_fn(samples):
    inputs = processor(samples)

    labels = inputs['input_ids'].clone()
    labels[:, :-1] = labels[:, 1:]
    padding_mask = (inputs['attention_mask'] == 0)
    labels[padding_mask] = -100
    inputs['labels'] = labels
    return inputs

def collate_fn_for_completion_only(samples, assistant_template='<|im_start|>assistant\n',chat_start_token_id=1):
    """
    only compatible with RLAIF-V dataset since it only has 1 conversation per chat.
    """
    inputs = processor(samples)
    labels = inputs['input_ids'].clone()
    labels[:, :-1] = labels[:, 1:].clone()
    padding_mask = (inputs['attention_mask'] == 0)
    labels[padding_mask] = -100
    inputs['labels'] = labels

    assistant_positions = []
    assistant_tokens = processor.tokenizer(assistant_template)['input_ids']

    for batch_idx in range(inputs['input_ids'].shape[0]):
        im_start_positions = torch.where(inputs['input_ids'][batch_idx]==chat_start_token_id)[0]
        for pos in im_start_positions:
            matched = False
            for i,token in enumerate(assistant_tokens):
                curr_pos = pos+i
                if inputs['input_ids'][batch_idx][curr_pos] != token:
                    break
            else:
                matched = True
        if matched:
            assistant_positions.append(pos)

    assert len(assistant_positions) == inputs['input_ids'].shape[0], "a sample in this batch doesn't contain the assistant_template"

    for batch_idx in range(inputs['input_ids'].shape[0]):
        inputs['labels'][batch_idx, :assistant_positions[batch_idx]-1] = -100

    return inputs


model = model.cuda()

dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = 8,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers = 6,
    pin_memory = True
)

def get_optimizer(model, vision_lr=1e-5, vision_proj_lr=1e-3, text_lr=1e-5):

    model._vision_trainable(trainable=False)
    model._text_trainable(trainable=False)

    vision_params = []
    vision_proj_params = []
    text_params = []

    for p in model.vision.parameters():
        vision_params.append(p)

    for p in model.vision_proj.parameters():
        p.requires_grad = True
        vision_proj_params.append(p)

    for n,p in model.text_model.named_parameters():
        text_params.append(p)

    param_groups = [
        {'params': vision_params, 'lr': vision_lr},
        {'params': vision_proj_params, 'lr': vision_proj_lr},
        {'params': text_params, 'lr': text_lr},
    ]

    optimizer = torch.optim.Adam(param_groups)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")

    return optimizer


losses = []
log_steps = 200
epochs = 3


unfreeze_text_pct = 0.
unfreeze_vision_pct = 0.

optim = get_optimizer(model,vision_proj_lr=1e-4 if unfreeze_vision_pct == 0 else 1e-3)
sched = torch.optim.lr_scheduler.OneCycleLR(
    optim,
    max_lr=[pg['lr'] for pg in optim.param_groups],
    total_steps=len(dataloader)*epochs,
    pct_start=0.25
)
print(optim)

global_step = 0
total_steps = len(dataloader)*epochs
checkpoint_steps = 1000
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

        global_step += 1
        if int(unfreeze_text_pct * total_steps) == global_step or (unfreeze_text_pct == 0 and global_step ==1):
            model._text_trainable(trainable=True)
            print('\n\n\t\tunfreezing text\n\n',sum(p.numel() for p in model.parameters() if p.requires_grad))
        if int(unfreeze_vision_pct * total_steps) == global_step or (unfreeze_text_pct == 0 and global_step ==1):
            model._vision_trainable(trainable=True)
            print('\n\n\t\tunfreezing vision\n\n',sum(p.numel() for p in model.parameters() if p.requires_grad))
        if global_step % checkpoint_steps == 0:
            sd = model.state_dict()
            torch.save(sd,'./Blinky/model-checkpoint-stage2.pt')
            gc.collect()
            torch.cuda.empty_cache()


fig=plt.figure()
plt.plot(losses)
plt.title('loss')
fig.savefig('losses_stage2.png')

sd = model.state_dict()
torch.save(sd,'./Blinky/stage2.pt')