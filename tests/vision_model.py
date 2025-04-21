import torch
import torch.nn as nn
from transformers import SiglipVisionModel
from einops import rearrange

class VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = SiglipVisionModel.from_pretrained('google/siglip2-base-patch16-512')
        self.shuffled_dim = self.vision.config.hidden_size * 4

    def pixel_shuffle(self, x):
        h = w = x.shape[1] ** 0.5
        s = 2 # scale_factor
        x = rearrange(x,'b (h w) d -> b h w d', h=w, w=w)
        x = rearrange(x,'b h (w_s s) d -> b h w_s (s d)',w_s=w//2,s=s)
        x = x.transpose(1,2) # b w_s h d*s
        x = rearrange(x,'b w_s (h_s s) d -> b w_s h_s (s d)',h_s=h//2,s=s)
        x = x.transpose(1,2) # b h_s w_s d*s*s
        x = x.flatten(1,2) # b t d*s*s 
        return x

    def forward(self, x):
        x = self.vision(x).last_hidden_state
        x = self.pixel_shuffle(x)
        return x


class VisionProjector(nn.Module):
    def __init__(self, vision_hidden_size, dim):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.dim = dim
        self.proj = nn.Linear(self.vision_hidden_size * 4, self.dim, bias=False)

    def forward(self, x):
        return self.proj(x)