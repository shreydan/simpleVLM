import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoTokenizer
from PIL import Image


def save_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_hf)
    tokenizer.save_pretrained('./Blinky')


class BlinkyProcessor:
    def __init__(self, tokenizer_path, num_image_tokens=196):
        self.tokenizer_path = tokenizer_path
        self.image_size = 224
        self.num_image_tokens = num_image_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        self.img_mean = [0.5,0.5,0.5]
        self.img_std = [0.5,0.5,0.5]
        self.img_transforms = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=self.img_mean, std=self.img_std)
        ])

    def preprocess_image(self, image):
        return self.img_transforms(image.convert('RGB'))

    def apply_chat_template(self, samples, use_system_prompt=True):
        chat_texts = self.tokenizer.apply_chat_template(
            samples,
            tokenize=False
        )
        return chat_texts

    def tokenize_and_pad(self, texts):
        tokenized = [self.tokenizer.encode(t,return_tensors='pt',truncation=True,max_length=1024).squeeze(0) for t in texts]
        max_length = max(t.shape[0] for t in tokenized)
        tokenized = [
            F.pad(t,[0,max_length-t.shape[0]],value=-100)
            for t in tokenized
        ] # right padding
        padded = torch.vstack(tokenized)
        attention_mask = torch.full_like(padded, 1)
        attention_mask[torch.where(padded==-100)] = 0
        padded[torch.where(padded==-100)] = self.tokenizer.pad_token_id
        return {
            'input_ids': padded,
            'attention_mask': attention_mask.bool()
        }

    def __call__(self, samples):
        texts = self.apply_chat_template([s['text'] for s in samples])
        inputs = self.tokenize_and_pad(texts)

        # inference
        if len(samples) == 1 and 'image' not in samples[0]:
            inputs['pixel_values'] = None
            return inputs

        images = torch.vstack([self.preprocess_image(s['image']).unsqueeze(0) for s in samples])
        inputs['pixel_values'] = images
        return inputs