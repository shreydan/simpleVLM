import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.vision_model import VisionEncoder
from text_model import LLaMA
from finetune_utils import update_embeddings, load_text_weights, Config


class Blinky(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_model = VisionEncoder(self.config.embed_dim).to(dtype=self.config.dtype)
        self.text_model = LLaMA(self.config)
        self.register_buffer('mask',self.prepare_prefix_lm_mask())

    def _tie_weights(self):
        self.text_model._tie_weights()

    def prepare_prefix_lm_mask(self):
        max_seq_len = self.config.max_position_embeddings
        prefix_len = self.config.prefix_length
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len),diagonal=1)
        # causal_mask[:prefix_len,:prefix_len] = 0
        return causal_mask

    def _prepare_for_training(self):
        self.text_model = load_text_weights(self.text_model)
        self.text_model = update_embeddings(
            self.text_model,
            old_vocab_size=self.config.vocab_size,
            new_vocab_size=self.config.vocab_size + 3,
            dim = self.config.embed_dim
        )

    def forward(self, input_ids, pixel_values=None, labels=None):

        text_embeddings = self.text_model.embed_tokens(input_ids)
        x = text_embeddings.clone()
        if pixel_values is not None:
            image_tokens = self.vision_model(pixel_values)
            for batch_idx in range(x.shape[0]):
                start_token_pos = torch.where(input_ids[batch_idx] == self.config.image_start_token_id)[0][0]+1
                end_token_pos = torch.where(input_ids[batch_idx] == self.config.image_end_token_id)[0][0]
                before = x[batch_idx, :start_token_pos, :]  # Tokens before image
                after = x[batch_idx, end_token_pos:, :]  # Tokens after image
                x[batch_idx] = torch.cat([before, image_tokens[batch_idx], after], dim=0)
                print(before.shape, after.shape, x[batch_idx].shape)


        assert torch.unique(x[:,0,:]) == 0, f'{torch.unique(x[:,0,:])}'
        assert torch.unique(x[:,end_token_pos,:]) == 0, f'{torch.unique(x[:,end_token_pos,:])}'

        print(start_token_pos, end_token_pos)
        assert torch.allclose(
            x[:,start_token_pos:end_token_pos,:],
            image_tokens,
        ), f'incorrect placement of image tokens. {torch.where(x[0,start_token_pos:end_token_pos,0]!=image_tokens[0,:,0])[0].cpu().numpy().tolist()}'

        for layer in self.text_model.layers:
            x = layer(x, self.mask)

        x = self.text_model.norm(x)
        logits = self.text_model.lm_head(x)

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss

        return logits


if __name__ == '__main__':
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
        dtype = torch.bfloat16,
        eos_token_id = 2,
        image_start_token_id = 49152,
        image_end_token_id = 49153,
        image_token_id = 49154,
        num_image_tokens = 256,
        prefix_length=1+256+1 #<|start_of_image|>256*<|image_token|><|end_of_image|>
    )
    model = SimpleVLM(config)
    model._prepare_for_training()
    input_ids = torch.randint(0,config.vocab_size-3,(1,300))
    input_ids[:,:config.prefix_length] = config.image_token_id
    inputs = {
        'input_ids': input_ids,
        'pixel_values': torch.rand(1,3,512,512)
    }
    outputs = model(**inputs)
    print(outputs.shape)
    print(model)