import torch
import torch.nn as nn
import torch.nn.functional as F
from text_model import LLaMA, SiLU
from types import SimpleNamespace
from transformers import SiglipVisionModel, SiglipVisionConfig
import gc


class VisionProjector(nn.Module):
    def __init__(self, vision_dim, proj_dim, dtype):
        super().__init__()
        self.embed_dim = vision_dim
        # self.intermediate_dim = self.embed_dim * 2
        self.proj_dim = proj_dim
        # self.gate_proj = nn.Linear(self.embed_dim, self.intermediate_dim, bias=False, dtype=dtype)
        # self.up_proj = nn.Linear(self.embed_dim, self.intermediate_dim, bias=False, dtype=dtype)
        # self.down_proj = nn.Linear(self.intermediate_dim, self.proj_dim, bias=False, dtype=dtype)
        # self.act_fn = SiLU()
        self.proj = nn.Linear(self.embed_dim, self.proj_dim, bias=False, dtype=dtype)

    def forward(self, x):
        # x [B S D]
        # x1 = self.gate_proj(x)
        # x2 = self.up_proj(x)
        # x = self.act_fn(x1) * x2
        # x = self.down_proj(x)
        x = self.proj(x)
        return x


class Blinky(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        vision_config = SiglipVisionConfig.from_pretrained(self.config.vision_model_hf)
        self.config.vision_config = SimpleNamespace(**vision_config.to_dict())

        self.vision = SiglipVisionModel(vision_config).to(dtype=self.config.dtype)

        self.vision_proj = VisionProjector(self.config.vision_config.hidden_size, self.config.embed_dim, dtype=self.config.dtype)
        self.text_model = LLaMA(self.config)

    def prepare_for_training(self):

        from transformers import SiglipVisionModel, AutoModelForCausalLM

        vision = SiglipVisionModel.from_pretrained(self.config.vision_model_hf, torch_dtype=self.config.dtype)
        self.vision.load_state_dict(vision.state_dict())

        assert torch.allclose(
            vision.vision_model.embeddings.position_embedding.weight,
            self.vision.vision_model.embeddings.position_embedding.weight
        ), 'couldnt load vision model'

        smol = AutoModelForCausalLM.from_pretrained(self.config.text_model_hf,torch_dtype=self.config.dtype)
        smol_sd = smol.state_dict()
        model_sd = self.text_model.state_dict()
        smol_sd = {k:v for k,v in smol_sd.items() if not any([s in k for s in ['rope','causal_mask']])}

        for smol_key,smol_value in smol_sd.items():
            model_key = smol_key.replace('model.','')
            model_sd[model_key] = smol_value.clone()

        self.text_model.load_state_dict(model_sd)

        assert torch.allclose(smol.lm_head.weight, self.text_model.lm_head.weight), 'couldnt load text model'

        del smol, vision
        gc.collect()

    def forward_image_features(self, pixel_values):
        x = self.vision(pixel_values).last_hidden_state
        x = self.vision_proj(x)
        return x

    def _vision_trainable(self,trainable=False):
        for p in self.vision.parameters():
            p.requires_grad=trainable

    def _text_trainable(self,trainable=False):
        for n,p in self.text_model.named_parameters():
            if 'embed_tokens' in n or 'lm_head' in n:
                p.requires_grad = trainable
            else:
                p.requires_grad = trainable

    def forward(self, input_ids, pixel_values=None, attention_mask=None, labels=None):

        x = self.text_model.embed_tokens(input_ids)

        image_tokens = None
        if pixel_values is not None:
            image_tokens = self.forward_image_features(pixel_values)
            x = torch.cat([image_tokens, x.detach()], dim=1)
            attention_mask = torch.cat([
                torch.full((x.shape[0],self.config.num_image_tokens),1).to(attention_mask.device).bool(),
                attention_mask
            ],dim=1)
            if labels is not None:
                labels = torch.cat([
                    torch.full((x.shape[0],self.config.num_image_tokens),-100).to(labels.device),
                    labels
                ],dim=1)

        for layer in self.text_model.layers:
            x = layer(x, attention_mask)

        x = self.text_model.norm(x)
        logits = self.text_model.lm_head(x)

        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            return loss

        return logits

if __name__ == '__main__':
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
    inputs = {
        'input_ids': torch.randint(0,config.vocab_size,(1,120)),
        'pixel_values': torch.rand(1,3,224,224),
    }
    outputs = model(**inputs)
    print(outputs.shape)