import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoTokenizer
from PIL import Image


class BlinkyProcessor:
    def __init__(self, tokenizer_path, num_image_tokens=256):
        self.tokenizer_path = tokenizer_path
        self.image_size = 512
        self.num_image_tokens = 256
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

    def _add_img_tokens(self, text):
        tokens = "<|start_of_image|><|image_token|><|end_of_image|>"
        tokens = tokens.replace("<|image_token|>","<|image_token|>"*self.num_image_tokens)
        return f"{tokens}\n{text}"
        
    def apply_chat_template(self, samples, use_system_prompt=True):
        chat_texts = self.tokenizer.apply_chat_template(
            samples, 
            tokenize=False
        )
        chat_texts = [
            self._add_img_tokens(chat_text)
            for chat_text in chat_texts
        ]
        return chat_texts

    def tokenize_and_pad(self, texts):
        tokenized = [self.tokenizer.encode(t,return_tensors='pt',truncation=True,max_length=1024).squeeze(0) for t in texts]
        max_length = max(t.shape[0] for t in tokenized)
        tokenized = [
            F.pad(t,[0,max_length-t.shape[0]],value=self.tokenizer.pad_token_id)
            for t in tokenized
        ] # right padding
        return torch.vstack(tokenized)

    def __call__(self, samples):
        texts = self.apply_chat_template([s['text'] for s in samples])
        input_ids = self.tokenize_and_pad(texts)
        images = torch.vstack([self.preprocess_image(s['image']).unsqueeze(0) for s in samples])
        return {
            'input_ids': input_ids,
            'pixel_values': images
        }