import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SiglipVisionModel

class RMSNorm(nn.Module):
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


class VisionEncoder(nn.Module):
    def __init__(self, dim, dtype):
        super().__init__()
        self.dim = dim
        self._vision_model_name = 'google/siglip-base-patch16-512'
        self.vision = SiglipVisionModel.from_pretrained(self._vision_model_name, torch_dtype=dtype)
        dtypes = []
        for p in self.vision.parameters():
            p.requires_grad = False
            dtypes.append(p.dtype)
        print(set(dtypes))

        self.num_tokens_per_image = self.vision.vision_model.embeddings.num_patches // 4

        config = self.vision.config
        self.patches_per_image = int(config.image_size // config.patch_size)
        self.tokens_per_side = int(self.num_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)
        self.rms_norm = nn.Identity() #RMSNorm(config.hidden_size)
        self.dim_proj = nn.Linear(config.hidden_size, self.dim, bias=False, dtype=dtype)

    def forward(self, x):
        if torch.isnan(x).any():
            print(f"NaN in INPUT IMAGE..")
        vision_outputs = self.vision(x).last_hidden_state
        if torch.isnan(vision_outputs).any():
            print(f"NaN in vision outputs..")
        batch_size, _, vision_dim = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, vision_dim, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)
        pooled_vision_outputs = self.rms_norm(pooled_vision_outputs)

        projected_vision_outputs = self.dim_proj(pooled_vision_outputs)
        if torch.isnan(projected_vision_outputs).any():
            print(f"NaN in projected vision outputs..")

        return projected_vision_outputs


if __name__ == '__main__':
    m = VisionEncoder(576)
    print(m(torch.rand(1,3,512,512)).shape)