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
        self.vision_model = VisionEncoder(self.config.embed_dim, dtype=self.config.dtype)
        self.text_model = LLaMA(self.config)

    def _tie_weights(self):
        self.text_model._tie_weights()

    def _prepare_for_training(self):
        self.text_model = load_text_weights(self.text_model)
        self.text_model = update_embeddings(
            self.text_model,
            old_vocab_size=self.config.vocab_size,
            new_vocab_size=self.config.vocab_size + 3,
            dim = self.config.embed_dim
        )

    def forward(self, input_ids, pixel_values=None, labels=None):
        # Get text embeddings
        if torch.isnan(input_ids).any():
            print(f"NaN in input-ids..")
        input_embeddings = self.text_model.embed_tokens(input_ids)  # [batch, seq_len, hidden_dim]
        if torch.isnan(input_embeddings).any():
            print(f"NaN after text embeddings..")
        if pixel_values is not None:
            # Compute image tokens
            image_tokens = self.vision_model(pixel_values)  # [batch, img_seq_len, hidden_dim]
            if torch.isnan(image_tokens).any():
                print(f"NaN in image embeddings..")
            x = input_embeddings.clone().detach()
            batch_size = x.shape[0]
            for i in range(batch_size):
                # Find start and end positions in each example.
                start_token_pos = (input_ids[i] == self.config.image_start_token_id).nonzero(as_tuple=True)[0][0] + 1
                end_token_pos = (input_ids[i] == self.config.image_end_token_id).nonzero(as_tuple=True)[0][0]

                # Check that the image token span matches the expected length.
                if (end_token_pos - start_token_pos) != image_tokens.shape[1]:
                    raise ValueError("Mismatch between token span length and image tokens length")

                x[i, start_token_pos:end_token_pos, :] = image_tokens[i]

        else:
            x = input_embeddings

        if torch.isnan(x).any():
            print(f"NaN after replacing text embeddings..")
        # Pass the (possibly modified) embeddings through the transformer layers.
        for i,layer in enumerate(self.text_model.layers):
            x = layer(x)
            if torch.isnan(x).any():
                print(f"NaN in layer {i}..")

        # Normalize and compute logits.
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